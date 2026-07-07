# Running bulk ingestion natively on a second machine (e.g. a 3080 Ti desktop)

Context: the laptop runs live campaigns via Docker (`docker compose up`) and
stays the one canonical place campaign/session data lives. A second machine
can be used purely as an offline corpus-building appliance — OCR + reindex +
lore/monster extraction — with no live play on it and (per this doc) no
Docker either, since Docker Desktop needs virtualization (WSL2/Hyper-V on
Windows) enabled, which this machine doesn't have.

**Read this before assuming `git checkout` gets you everything**: `docs/raw/`,
`docs/source/`, and `data/` are all in `.gitignore` (raw PDFs are large;
`docs/source/` is OCR'd copyrighted WotC text, not redistributed via git;
`data/` is DB/embedding output). So `git clone`/`checkout` only gets you
**code** (Makefile, scripts, backend/). The actual PDFs, OCR'd markdown,
entity JSON, and ChromaDB/BM25 data all need a **separate manual transfer**
(external drive, rsync over LAN, cloud storage — whatever's convenient), in
both directions:
- **Laptop → desktop**: the raw PDFs (`docs/raw/done/*.pdf`) need to physically
  get to the desktop before it can OCR anything.
- **Desktop → laptop**: the resulting `docs/source/**/*.md`, the
  `_entities.json`/`_entities/*.json` registries, `data/chroma_db/`, and
  `data/bm25_rules.pkl` need to come back — see "Syncing back," below.

## One-time desktop setup

1. `git clone`/checkout the repo (gets Makefile, scripts, requirements.txt).
2. Copy the PDFs you want processed into `docs/raw/` on the desktop (any
   subset — you don't need the whole library at once).
3. **Python venv** (cross-platform target, works on Windows too):
   ```
   make setup-venv
   ```
   Installs the exact same `requirements.txt` the Docker image uses —
   everything `build_index.py`/`extract_entities.py`/`merge_chroma.py`/
   `load_lore_json.py` need. This is a completely separate install from OCR.
4. **OCR (MinerU) — separate install, platform-specific backend.** On the
   laptop this uses MinerU's `mlx-engine` (Apple Silicon/Metal). On an Nvidia
   desktop it needs MinerU's CUDA path instead:
   ```
   .venv/Scripts/pip install -U "mineru[all]"     # Windows
   ./.venv/bin/pip install -U "mineru[all]"       # Linux
   ```
   **I haven't verified this on real CUDA hardware** (this was built/tested
   on the Apple Silicon laptop only) — the analogous check to the laptop's
   `_select_mac_engine()` would be confirming `mineru`/the underlying `torch`
   install actually detects the 3080 Ti (`python -c "import torch;
   print(torch.cuda.is_available())"` should print `True`; if it prints
   `False`, you have a CPU-only torch wheel installed and need the CUDA
   build from pytorch.org matching your CUDA version first). Do this check
   — and a small one-page OCR smoke test — before trusting it for a
   real overnight run, same discipline as the laptop's own MLX verification.
5. **Ollama, with a model that fits 12GB VRAM.** The app's default
   (`settings.mechanics_model`, `gemma4:26b-mlx`) is a non-starter here twice
   over: it's ~26B (won't fit 12GB even quantized) and it's in MLX format,
   which is Apple-only — Ollama on Nvidia uses GGUF/CUDA, so this exact tag
   can't load at all.

   **Target model: `gemma4:e4b`** — same vendor family as the already-trusted
   default (`gemma4:26b-mlx`), an efficient/smaller-footprint variant sized
   to actually fit 12GB VRAM. Deliberately NOT `qwen2.5:14b` despite being
   roughly the right size class — `design.md` documents a real, serious
   incident with that exact model in this codebase (fake tool-call blocks,
   garbled output under sustained use during Session 0/world-prep). That
   failure was in a multi-turn tool-calling agent loop, a different task
   shape than the single-shot completions contextualization/extraction use
   here, so it may not directly apply — but no reason to inherit that risk
   when a same-family alternative is the target anyway. **Smoke-test it**
   (a handful of chunks) before trusting it for an unattended multi-hour
   run — this codebase's own established discipline, not new caution
   invented for this doc.
   ```
   ollama pull gemma4:e4b
   ```

## Running ingestion

No Postgres needed on this machine at all — `build_index.py` never touches
it, and `extract_entities.py` defaults to JSON-only here (see below).

```
make ingest-book-native adventure="Curse of Strahd" context_model=gemma4:e4b model=gemma4:e4b
make ingest-book-native book="D&D 5E - Dungeon Master's Guide" source_type=core context_model=gemma4:e4b model=gemma4:e4b
make ingest-book-native book="D&D 5E - Volo's Guide to Monsters" source_type=core kinds=monster context_model=gemma4:e4b model=gemma4:e4b
```

Same resumability guarantees as the Docker version (both scripts are
identical — only the runner differs): OCR-side chunking/checkpointing is in
`ocr_ingest.py` itself (run that separately first, same as on the laptop —
`ingest-book-native` starts from already-OCR'd `docs/source/`), and
`build_index.py`/`extract_entities.py`'s own per-batch/per-entity
checkpointing is unchanged.

`extract_entities.py` writes to Postgres only if you explicitly pass
`write_postgres=1` (and have a native Postgres reachable — not set up by
anything in this doc, since it's simpler to just sync the JSON back and load
it on the laptop instead — see next section).

## Syncing back to the laptop

Copy (rsync/external drive/etc.) these from the desktop back to the laptop,
same relative paths:
- `docs/source/` (the OCR'd markdown + `_entities.json`/`_entities/*.json`
  registries)
- `data/chroma_db/`
- `data/bm25_rules.pkl`

Then, on the laptop:

```
make merge-chroma source=/path/to/copied/chroma_db
make load-lore-json all_core=1          # or --book "Name" for just one
make load-lore-json all_adventures=1
```

`merge-chroma` streams the desktop's Chroma collection into the laptop's
canonical one page-by-page (deliberately small pages — see
`scripts/merge_chroma.py`'s own comments for why: a naive full-corpus buffer
OOM-killed this app's Docker container even though the corpus itself wasn't
that large, because this whole Docker Desktop VM has under 1GB total,
shared with the live app server) and rebuilds the BM25 pickle from the
merged result. `load-lore-json` upserts the JSON registries into Postgres
without re-running any LLM extraction. Both are safe to re-run — re-syncing
the desktop's output after it's done more books overnight and re-running
these two commands just picks up whatever's new (idempotent upsert, keyed
on deterministic content-derived ids for Chroma, `(book_slug, entity_type,
canonical_name)` for lore entities).

## What NOT to do

Don't run live campaigns on the desktop. There's no natural way to merge two
independently-played campaigns' Postgres state, unlike the corpus data above
(which merges cleanly because it's all either content-addressed or has a
real uniqueness constraint). This machine's job is bulk offline corpus
building only.
