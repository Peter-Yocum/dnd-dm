# Nightly per-book ingestion commands

One book at a time, run whenever you like — no rush, no dependency between
books. Each book is ONE command: `make ingest-book`, which runs two steps in
sequence (both show a live `tqdm` progress bar):

1. **Reindex with contextualization** — `build_index.py`, scoped to just this
   book. Deletes only this book's existing chunks, rebuilds them under the
   parent/child + contextual-augmentation schema. Every other book's chunks
   are untouched — search keeps working for everything else while this runs.
2. **Extract lore/monsters** — `extract_entities.py --write-postgres`, scoped
   to the same book. Populates the canon Lore Registry (NPCs, locations,
   items, and — for books with real stat blocks — monsters).

Both steps are resumable: safe to Ctrl-C or kill (laptop sleep, crash,
whatever) and re-run the exact same command — already-completed work is
skipped, not redone from scratch.

**No `--wipe` needed anywhere here.** Going through every book this way
eventually migrates the whole corpus to the new retrieval schema without
ever blacking out search for books not yet touched — safer than a single
big `--wipe` rebuild, given there's a real live campaign ("Out of the Abyss,
into the Fire") that might get played in between nights.

**Timing**: contextualization is one local LLM call per child chunk. This
varies a lot by book size — core rulebooks and the bigger adventures (Curse
of Strahd alone is ~150K parent+child chunks) will likely take multiple
nights each; something Lost Mine of Phandelver-sized (~6-7K chunks) should
be much faster. I haven't precisely measured per-chunk latency, so treat
these as genuinely open-ended overnight jobs, not a fixed ETA — watch the
progress bar's rate/ETA once it's been running a few minutes for a real
estimate.

**Desktop GPU (3080 Ti, 12GB VRAM)**: the main 26B mechanics model won't
fit, but contextualization is a much lighter single-shot task that would —
once Ollama is running on that machine with a smaller model (7-8B class)
pulled, these commands would need an `--ollama-url http://<desktop-ip>:11434`
override (already supported by the underlying scripts, not yet exposed as a
Makefile param). There's also currently no flag to pick a *different* model
than the main 26B one for just this pass (it always uses
`settings.mechanics_model`) — say the word if you want both added before
setting the desktop up.

---

## Tonight: Lost Mine of Phandelver

Already OCR'd and fast-indexed (no contextualization) for testing. Re-run
without `skip_context` to get the real contextualized version — this will
delete the fast-indexed chunks for this adventure and rebuild them properly.

```bash
make ingest-book adventure="Lost Mine of Phandelver"
```

---

## Adventures

```bash
make ingest-book adventure="Curse of Strahd"
make ingest-book adventure="Ghosts of Saltmarsh"
make ingest-book adventure="Icewind Dale"
make ingest-book adventure="Storm King's Thunder"
make ingest-book adventure="Tales of the Yawning Portal"
make ingest-book adventure="Tomb of Annihilation"
make ingest-book adventure="Tyranny of Dragons"
make ingest-book adventure="Waterdeep"
```

### Out of the Abyss — SKIP for now
Its `docs/source/adventures/Out of the Abyss/` folder is currently empty
(confirmed: 0 chunks) — the PDF is queued for re-OCR (see
`docs/engineering-notes/reingest-followup-prompt.md`) but hasn't been
re-ingested yet. Running the command above for it right now would be a
no-op.

---

## Core rulebooks

Core books need `book="<exact filename stem>" source_type=core` instead of
`adventure=` (this is new plumbing added specifically for this —
`build_index.py` previously could only do all-core-at-once). The **Monster
Manual** gets `kinds=monster`, since it's almost entirely stat blocks
(confirmed: it's one of the largest core books, ~31.5K parent+child chunks)
— running full NPC/location/item discovery on it would burn a lot of LLM
calls finding nothing. Every other core book keeps the default (all four
kinds), since the PHB/DMG/setting guide do have real named NPCs/places in
their fiction and worldbuilding sections.

```bash
make ingest-book book="D&D 5.5E - Player's Handbook" source_type=core
make ingest-book book="D&D 5E - Dungeon Master's Guide" source_type=core
make ingest-book book="D&D 5E - Monster Manual" source_type=core kinds=monster
make ingest-book book="D&D 5E - Mordenkainen's Tome of Foes" source_type=core
make ingest-book book="D&D 5E - Sword Coast Adventurer's Guide" source_type=core
make ingest-book book="D&D 5E - Tasha's Cauldron of Everything" source_type=core
make ingest-book book="D&D 5E - Volo's Guide to Monsters" source_type=core
make ingest-book book="D&D 5E - Xanathar's Guide to Everything" source_type=core
```

---

## After a book's Lore Registry is populated (optional, campaign-specific)

Two more steps exist but are **per-campaign, not per-book** — run them for
the real "Out of the Abyss, into the Fire" campaign (or any campaign)
whenever you want its already-created NPCs/Locations/Items linked back to
whatever canon data now exists for its books:

```bash
make backfill-lore-links campaign_id=<uuid> dry_run=1   # review first
make backfill-lore-links campaign_id=<uuid>              # then for real
make seed-relation-graph campaign_id=<uuid>               # relation graph, no LLM calls
```
