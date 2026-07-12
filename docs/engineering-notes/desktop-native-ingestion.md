# Running bulk ingestion natively on a second machine (e.g. a 3080 Ti desktop)

> **2026-07-12: `merge-chroma`/`scripts/merge_chroma.py` referenced below are
> obsolete** — the rules corpus moved from ChromaDB (a local file store) to
> Postgres/pgvector (a real networked DB). If this machine can reach the
> canonical machine's Postgres (same LAN, port 5432), just point `DATABASE_URL`
> at it and run `ingest-book-native` directly — no merge step needed. See the
> Makefile's `merge-chroma` target comment. The rest of this doc (OCR,
> extraction, the no-Docker/no-virtualization rationale) is otherwise still
> accurate.

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

## One-time desktop setup (Windows / PowerShell, no `make`)

The desktop has no `make` available (and no virtualization, so no Docker
either — everything below is plain PowerShell + a native Python venv). The
Makefile targets mentioned later in this doc (`merge-chroma`,
`load-lore-json`) run on the **laptop** (Mac, which has `make`) — this
section is desktop-only and deliberately spells out raw commands instead.

1. `git clone`/checkout the repo (gets Makefile, scripts, requirements.txt —
   `make` isn't needed to just have the files, only to *run* targets).
2. Copy the PDFs you want processed into `docs\raw\` on the desktop (any
   subset — you don't need the whole library at once).
3. **Python venv:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   Installs the exact same `requirements.txt` the Docker image uses —
   everything `build_index.py`/`extract_entities.py`/`merge_chroma.py`/
   `load_lore_json.py` need. This is a completely separate install from OCR.
   (If a later Mac/Linux machine ever does this instead, `make setup-venv`
   is the equivalent one-liner there, since `make` exists on those.)
4. **OCR (MinerU) — separate install, platform-specific backend.** On the
   laptop this uses MinerU's `mlx-engine` (Apple Silicon/Metal). On an Nvidia
   desktop it needs MinerU's CUDA path instead:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -U "mineru[all]"
   ```
   **I haven't verified this on real CUDA hardware** (this was built/tested
   on the Apple Silicon laptop only) — the analogous check to the laptop's
   `_select_mac_engine()` would be confirming `mineru`/the underlying `torch`
   install actually detects the 3080 Ti:
   ```powershell
   .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True`; if it prints `False`, you have a CPU-only torch
   wheel installed and need the CUDA build from pytorch.org matching your
   CUDA version first. Do this check — and a small one-page OCR smoke test
   (`--pages 1`) — before trusting it for a real overnight run, same
   discipline as the laptop's own MLX verification.
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

## Running ingestion (PowerShell)

No Postgres needed on this machine at all — `build_index.py` never touches
it, and plain `extract_entities.py` (no `--write-postgres`) writes only its
JSON registry.

Each book is OCR, then reindex, then extract — three separate commands
(the `make ingest-book-native` target on a `make`-capable machine bundles
the last two; here they're just spelled out):

**Known Windows quirk, confirmed live:** a fresh `pip install -U "mineru[all]"`
on Windows can resolve a newer MinerU release than the Mac's (3.4.2) — its
`-b`/`--backend` flag has different valid choices (`vlm-auto-engine`/
`hybrid-auto-engine` instead of `vlm-engine`/`hybrid-engine`). If OCR fails
with `invalid value for -b`, run `mineru --help`, check the `-b` line's
actual choices, and pass the right one via `--mineru-backend`:

```powershell
# 1. OCR (once per book — skips already-OCR'd books unless --force)
.\.venv\Scripts\python.exe scripts\ocr_ingest.py --file "docs\raw\done\D&D 5E - Dungeon Master's Guide.pdf" --output docs\source\core --mineru-backend vlm-auto-engine

# 2. Reindex (contextualize + embed + BM25)
.\.venv\Scripts\python.exe scripts\build_index.py --book "D&D 5E - Dungeon Master's Guide" --source-type core --context-model gemma4:e4b

# 3. Extract lore/monsters (JSON only — no --write-postgres, no Postgres on this machine)
.\.venv\Scripts\python.exe scripts\extract_entities.py --book "D&D 5E - Dungeon Master's Guide" --source-type core --model gemma4:e4b
```

For an adventure instead of a core book, swap `--book "..." --source-type core`
for `--adventure "Curse of Strahd"` on the `build_index.py` call, and
`--book "Curse of Strahd"` (default `--source-type adventure`) on
`extract_entities.py`. For the Monster Manual specifically, add
`--kinds monster` to the `extract_entities.py` call (see
`nightly-ingestion-commands.md` for why).

Same resumability guarantees as the Docker version (identical scripts, only
the runner differs): `ocr_ingest.py`'s own per-window chunking/checkpointing,
and `build_index.py`/`extract_entities.py`'s per-batch/per-entity
checkpointing, are all unchanged — safe to Ctrl-C and re-run any of the
three commands above.

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
