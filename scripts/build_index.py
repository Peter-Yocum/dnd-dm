#!/usr/bin/env python3
"""
build_index.py — Chunk, contextualize, and embed docs/source/ into Postgres
(rule_chunks, pgvector + native full-text search) for hybrid RAG, with a
parent/child chunk split.

Expected folder structure:
    docs/source/core/                       ← searched for every campaign
    docs/source/adventures/{slug}/          ← searched only for campaigns that include {slug}
    docs/source/adventures/{slug}/_meta.json  ← optional display metadata

Each .md file is split on ## headers into PARENT sections (as before), and
each parent is further split into smaller CHILD sub-chunks. Only children are
embedded (dense, `embedding` column) and keyword-indexed (`content_tsv`,
generated automatically — see backend/stores/tables.py); parents exist
purely as an id-addressable lookup target so a search hit can be expanded to
its full surrounding section (backend/stores/rules_store.py's search()).
Every child chunk is contextualized before embedding — a short LLM-generated
blurb is prepended to the text that gets EMBEDDED, but the chunk's stored/
citable text (`content` in rule_chunks, `.content` on RuleChunk) stays the
raw, unmodified book text.

Metadata per chunk:
    book             — filename stem formatted as title
    section          — ## header text
    source_type      — "core" | "adventure"
    adventure        — slug (empty string for core)
    granularity      — "parent" | "child"
    parent_chunk_id  — (child only) the id of its parent document
    doc_id           — stable per-book identity (source_type::adventure::book)
    sequence_number  — this chunk's parent section's ordinal within its book

Resumability (safe to Ctrl-C/kill and re-run without losing completed work
or redoing it): before any embedding/contextualization work for a given
chunk_id, this script checks whether that id already exists in rule_chunks
and skips it if so (unless --force). The table itself IS the completion
tracker — no second cache file to keep in sync. Chunks are embedded and
upserted in small batches (not buffered for a whole book), so a kill loses
at most one small in-flight batch.

IMPORTANT — --wipe/--fresh and resuming after a kill: a scoped run
(--book/--adventure/--source-type) is INCREMENTAL by default — existing rows
in that scope are left alone, only missing chunk_ids get added, so re-running
the same scoped command after a crash is cheap and safe. --wipe (whole
collection) and --fresh (just the current scope) both delete existing rows
FIRST, ONCE, before indexing proceeds with the resumable behavior above — use
either only for a genuine rebuild (e.g. after a chunking-schema change), never
to resume an interrupted run, since repeating either would destroy progress
already saved. See `make index` vs `make reindex-full` in the Makefile.

Usage:
    python build_index.py                        # index everything (resumable)
    python build_index.py --wipe                  # full rebuild only: clear collection, then index
    python build_index.py --source-type core      # core books only (incremental)
    python build_index.py --adventure tyranny-of-dragons  # one adventure only (incremental)
    python build_index.py --adventure tyranny-of-dragons --fresh  # same, but wipe that
                                                    # adventure's rows first (schema change, etc.)
    python build_index.py --skip-contextualization  # fast dev path, no LLM calls
    python build_index.py --force                 # re-process even already-indexed chunk_ids
    python build_index.py --recontextualize        # resumable: (re-)contextualize only rows
                                                    # indexed with contextualized=false so far
                                                    # (e.g. from a prior --skip-contextualization
                                                    # run) — safe to Ctrl-C/kill and rerun

Run inside the app container via:  make index  /  make reindex-full
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Windows' default console codec (cp1252/"charmap") can't encode the em-dashes/
# box-drawing/arrow characters this script prints for readability — confirmed
# live as a real UnicodeEncodeError crash on a fresh Windows venv (Mac/Linux
# default to UTF-8 stdout so this never surfaced there). errors="replace"
# rather than "strict" so a genuinely unencodable character degrades to "?"
# instead of crashing an otherwise-successful chunk mid-run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow running as `python scripts/build_index.py` from the repo root —
# Python sets sys.path[0] to this script's own directory (scripts/), not the
# repo root, so `backend` wouldn't otherwise be importable by the lazy
# `from backend.rag...` imports below (same fix as extract_entities.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SOURCE  = "docs/source"
DEFAULT_OLLAMA  = "http://localhost:11434"
DEFAULT_VLLM    = "http://localhost:8100/v1"  # vllm-metal chat server, see vllm-migration-plan.md
DEFAULT_EMBED   = "nomic-embed-text"
MAX_CHUNK_CHARS = 1500   # parent section cap
CHILD_CHUNK_CHARS = 350  # child sub-chunk target size (dense/BM25 search target)
CHUNK_OVERLAP_WORDS = 50 # trailing words carried into the next size-split part
INDEX_BATCH_SIZE = 8     # chunks embedded/upserted per batch — small, so a kill loses little

# ── chunking ──────────────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)


def _split_words_with_overlap(text: str, max_chars: int, overlap_words: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Word-accumulate text into ~max_chars pieces, carrying the trailing
    overlap_words words of each piece into the next one — so a reader (or an
    embedding) never loses continuity right at a chunk boundary. Used both
    for splitting an oversized parent section and for splitting a parent
    into children."""
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    chunk: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > max_chars and chunk:
            pieces.append(" ".join(chunk))
            overlap = chunk[-overlap_words:]
            chunk = list(overlap)
            length = sum(len(w) + 1 for w in chunk)
        chunk.append(word)
        length += len(word) + 1
    if chunk:
        pieces.append(" ".join(chunk))
    return pieces


def chunk_markdown(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str, str]]:
    """Split markdown into (section_title, parent_chunk_text) pairs on
    ## / ### headers. An oversized section is split by size using
    _split_words_with_overlap (fixes a real bug: the previous version had
    zero overlap between size-split parts, losing continuity right at the
    seam)."""
    matches = list(_HEADER_RE.finditer(text))
    sections: list[tuple[str, str]] = []

    def _add(title: str, body: str) -> None:
        body = body.strip()
        if not body:
            return
        if len(body) <= max_chars:
            sections.append((title, f"## {title}\n\n{body}"))
            return
        pieces = _split_words_with_overlap(body, max_chars)
        for part, piece in enumerate(pieces, start=1):
            label = f"{title} ({part})" if len(pieces) > 1 else title
            sections.append((label, f"## {label}\n\n{piece}"))

    if not matches:
        _add("Document", text)
        return sections

    preamble = text[:matches[0].start()].strip()
    if preamble:
        _add("Introduction", preamble)

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        _add(title, text[start:end])

    return sections


# ── document builder ──────────────────────────────────────────────────────────

def _chunk_id(source_type: str, adventure: str, book: str, section: str, sample: str, ordinal: int, part: str) -> str:
    """part discriminates a parent id from its children's ids (e.g. "parent",
    "child0", "child1", ...) so they never collide despite sharing the same
    book/section/ordinal."""
    key = f"{source_type}::{adventure}::{book}::{section}::{sample}::{ordinal}::{part}"
    return hashlib.md5(key.encode()).hexdigest()


def build_documents(source_dir: Path) -> list[dict]:
    """Walk source_dir and return a list of {id, text, meta} dicts — one
    PARENT doc plus N CHILD docs per parent section. `text` is always the
    raw, citable book text (never a contextualization blurb)."""
    docs: list[dict] = []
    core_dir = source_dir / "core"
    adv_dir  = source_dir / "adventures"

    def _emit(book_name: str, text: str, source_type: str, adventure: str) -> int:
        doc_id = f"{source_type}::{adventure}::{book_name}"
        count = 0
        for ordinal, (section, parent_text) in enumerate(chunk_markdown(text)):
            parent_id = _chunk_id(source_type, adventure, book_name, section, parent_text[:32], ordinal, "parent")
            base_meta = {
                "book": book_name, "section": section,
                "source_type": source_type, "adventure": adventure,
                "doc_id": doc_id, "sequence_number": ordinal,
            }
            # chunk_id is duplicated into meta (not just used as the "id"
            # key/primary key) since _index_documents reads it straight out
            # of meta when building each row for rule_chunks — keeps the dict
            # self-contained rather than needing the caller to zip "id" back
            # in separately.
            docs.append({
                "id": parent_id, "text": parent_text,
                "meta": {**base_meta, "granularity": "parent", "chunk_id": parent_id},
            })
            count += 1
            children = _split_words_with_overlap(parent_text, CHILD_CHUNK_CHARS)
            for child_ordinal, child_text in enumerate(children):
                child_id = _chunk_id(
                    source_type, adventure, book_name, section, child_text[:32], ordinal, f"child{child_ordinal}",
                )
                docs.append({
                    "id": child_id, "text": child_text,
                    "meta": {**base_meta, "granularity": "child", "parent_chunk_id": parent_id, "chunk_id": child_id},
                })
                count += 1
        return count

    if core_dir.exists():
        before = len(docs)
        for md_file in sorted(core_dir.glob("*.md")):
            book_name = md_file.stem.replace("-", " ").replace("_", " ").title()
            _emit(book_name, md_file.read_text(encoding="utf-8"), "core", "")
        print(f"  core/  — {len(docs) - before} chunks (parent+child)")

    if adv_dir.exists():
        for adv_folder in sorted(adv_dir.iterdir()):
            if not adv_folder.is_dir() or adv_folder.name.startswith("_"):
                continue
            slug = adv_folder.name
            meta_file = adv_folder / "_meta.json"
            adv_name  = (json.loads(meta_file.read_text())["name"]
                         if meta_file.exists() else slug.replace("-", " ").title())
            before = len(docs)
            for md_file in sorted(adv_folder.glob("*.md")):
                book_name = md_file.stem.replace("-", " ").replace("_", " ").title()
                _emit(book_name, md_file.read_text(encoding="utf-8"), "adventure", slug)
            print(f"  adventures/{slug}/  ({adv_name}) — {len(docs) - before} chunks (parent+child)")

    return docs


# ── Postgres helpers (2026-07-12: replaced ChromaDB — see design.md) ─────────
# content_tsv (keyword search) is a GENERATED ALWAYS AS column on rule_chunks
# (see alembic/versions/0004_rules_and_chronicles_pgvector.py) — always in
# sync with `content` automatically, so there is no separate rebuild step
# here the way there used to be for the old BM25/FTS5 sidecar files.

def _get_engine():
    from sqlalchemy import create_engine

    from backend.config import settings
    return create_engine(settings.database_url)


def _wipe_table(engine) -> None:
    from backend.stores import tables as t

    with engine.begin() as conn:
        conn.execute(t.rule_chunks.delete())
    print("  Wiped rule_chunks table.")


def _delete_where(engine, condition) -> None:
    from backend.stores import tables as t

    with engine.begin() as conn:
        conn.execute(t.rule_chunks.delete().where(condition))


# ── indexing (resumable) ──────────────────────────────────────────────────────

def _index_documents(
    all_docs: list[dict], engine, ollama_url: str, vllm_url: str,
    skip_contextualization: bool, force: bool, context_model: str | None = None,
    recontextualize: bool = False,
) -> None:
    from tqdm import tqdm
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from backend.llm import ollama_embeddings
    from backend.rag.contextualizer import ChunkContextualizer
    from backend.stores import tables as t

    # Embeddings stay on Ollama (ollama_url) — the vllm-metal migration's
    # embeddings step (vllm-migration-plan.md §7.7) is separate/not done
    # yet. Contextualization is a chat call and moved to vLLM (vllm_url)
    # with the rest of the app's chat traffic.
    embeddings_fn = ollama_embeddings(model=DEFAULT_EMBED, base_url=ollama_url, timeout=None)
    if skip_contextualization:
        contextualizer = None
    elif context_model:
        contextualizer = ChunkContextualizer(model=context_model, vllm_base_url=vllm_url)
    else:
        contextualizer = ChunkContextualizer(vllm_base_url=vllm_url)

    total = len(all_docs)
    indexed_count = 0
    skipped_count = 0
    progress = tqdm(total=total, unit="chunk", desc="Indexing", dynamic_ncols=True)
    for i in range(0, total, INDEX_BATCH_SIZE):
        batch = all_docs[i:i + INDEX_BATCH_SIZE]
        batch_ids = [d["id"] for d in batch]

        if not force:
            # --recontextualize resumability: a row that already exists but
            # was indexed with contextualized=false (e.g. a prior
            # --skip-contextualization run) still needs reprocessing — only
            # skip rows that are already contextualized. A plain run just
            # skips anything that exists at all, same as before.
            with engine.connect() as conn:
                if recontextualize:
                    existing = {
                        row.chunk_id for row in conn.execute(
                            select(t.rule_chunks.c.chunk_id).where(
                                t.rule_chunks.c.chunk_id.in_(batch_ids),
                                t.rule_chunks.c.contextualized.is_(True),
                            )
                        )
                    }
                else:
                    existing = {
                        row.chunk_id for row in conn.execute(
                            select(t.rule_chunks.c.chunk_id).where(t.rule_chunks.c.chunk_id.in_(batch_ids))
                        )
                    }
            batch = [d for d in batch if d["id"] not in existing]

        if batch:
            embed_texts = []
            for d in batch:
                if d["meta"]["granularity"] == "child" and contextualizer is not None:
                    # find this child's parent text for situating context —
                    # parent_text is only available at build_documents() time,
                    # so a lightweight cache is passed via d["_parent_text"].
                    blurb = contextualizer.contextualize(
                        d["text"], d.get("_parent_text", d["text"]), d["meta"]["book"],
                    )
                    embed_texts.append(f"{blurb}\n\n{d['text']}" if blurb else d["text"])
                else:
                    embed_texts.append(d["text"])

            # Parent rows are never embedded/searched directly — no dense
            # step needed for them, only children.
            children_idx = [j for j, d in enumerate(batch) if d["meta"]["granularity"] == "child"]
            child_vectors = embeddings_fn.embed_documents([embed_texts[j] for j in children_idx]) if children_idx else []
            vectors_by_idx = dict(zip(children_idx, child_vectors))

            rows = []
            for j, d in enumerate(batch):
                meta = d["meta"]
                is_child = meta["granularity"] == "child"
                rows.append({
                    "chunk_id": d["id"],
                    "book": meta["book"],
                    "section": meta["section"],
                    "source_type": meta["source_type"],
                    "adventure": meta.get("adventure", ""),
                    "granularity": meta["granularity"],
                    "parent_chunk_id": meta.get("parent_chunk_id", ""),
                    "sequence_number": meta.get("sequence_number", 0),
                    "content": d["text"],
                    "embedding": vectors_by_idx.get(j),
                    # Parent rows are never contextualized (never embedded/
                    # searched directly), so they're trivially "done" and
                    # shouldn't be picked up by a --recontextualize pass.
                    "contextualized": (contextualizer is not None) if is_child else True,
                })

            stmt = pg_insert(t.rule_chunks).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["chunk_id"],
                set_={
                    "content": stmt.excluded.content,
                    "embedding": stmt.excluded.embedding,
                    "book": stmt.excluded.book,
                    "section": stmt.excluded.section,
                    "source_type": stmt.excluded.source_type,
                    "adventure": stmt.excluded.adventure,
                    "granularity": stmt.excluded.granularity,
                    "parent_chunk_id": stmt.excluded.parent_chunk_id,
                    "sequence_number": stmt.excluded.sequence_number,
                    "contextualized": stmt.excluded.contextualized,
                },
            )
            with engine.begin() as conn:
                conn.execute(stmt)

        batch_size = len(batch_ids)
        indexed_count += len(batch)
        skipped_count += batch_size - len(batch)
        progress.set_postfix(indexed=indexed_count, skipped=skipped_count)
        progress.update(batch_size)
    progress.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the hybrid Postgres/pgvector rules index from docs/source/.")
    ap.add_argument("--source",      default=DEFAULT_SOURCE)
    ap.add_argument("--ollama-url",  default=None, help="Ollama server for embeddings")
    ap.add_argument("--vllm-url",    default=None,
                     help="vLLM-metal chat server for contextualization (see "
                          "vllm-migration-plan.md) — separate from --ollama-url, "
                          "which is embeddings-only now")
    ap.add_argument("--wipe",        action="store_true",
                     help="clear whole collection first — FIRST RUN ONLY; if interrupted, "
                          "resume with a plain run (no --wipe), not by repeating --wipe")
    ap.add_argument("--source-type", choices=["core", "adventure"], default=None)
    ap.add_argument("--adventure",   default=None, help="index one adventure slug only")
    ap.add_argument("--book",        default=None,
                     help="index one core rulebook only, by its raw filename stem "
                          "(same value you'd pass to extract_entities.py --book --source-type core), "
                          "e.g. --book \"D&D 5E - Monster Manual\". Adventures already have "
                          "per-adventure granularity via --adventure; this fills the same gap for core/.")
    ap.add_argument("--fresh",       action="store_true",
                     help="delete this run's scoped existing chunks (--book/--adventure/"
                          "--source-type core) before reindexing — use after a chunking-schema "
                          "change. Default is incremental: existing chunk_ids are left alone and "
                          "only missing ones get added, so re-running the same scoped command (e.g. "
                          "to resume after a crash) is cheap and doesn't refetch/re-embed "
                          "everything. Requires one of --book/--adventure/--source-type — for a "
                          "full unscoped rebuild use --wipe instead. (2026-07-13: a scoped run "
                          "used to delete its scope's existing rows unconditionally, with no way "
                          "to opt out — this flag makes that an explicit choice instead of the "
                          "default, after an incident where a scoped run silently found zero docs "
                          "to reindex, due to an unrelated Docker mount gap, and deleted a whole "
                          "book's rows with nothing to replace them.)")
    ap.add_argument("--skip-contextualization", action="store_true",
                     help="fast dev path: skip the LLM contextualization pass entirely")
    ap.add_argument("--context-model", default=None,
                     help="vLLM-served model name for the contextualization pass (default: "
                          "settings.mechanics_model, mlx-community/Qwen3-30B-A3B-4bit as of the "
                          "2026-07-13 vllm-metal migration — see vllm-migration-plan.md; served via "
                          "--vllm-url, NOT an Ollama model tag anymore). On a machine that can't run "
                          "the default (no vllm-metal set up, or a smaller-GPU machine), pass a "
                          "different model served by whatever's at --vllm-url explicitly rather than "
                          "silently falling back to it — this codebase has a documented incident "
                          "(design.md) where an unvalidated smaller model (qwen2.5:14b) produced fake "
                          "tool calls/garbled output under sustained use, so prefer an already-validated "
                          "model family and smoke-test before trusting a substitute for a long "
                          "unattended run. NOTE: docs/engineering-notes/desktop-native-ingestion.md's "
                          "native-desktop workflow predates this migration and still assumes Ollama for "
                          "this pass — needs its own follow-up update to set up vllm-metal there too, "
                          "not covered by this change.")
    ap.add_argument("--force",       action="store_true",
                     help="re-process chunk_ids that already exist in the collection")
    ap.add_argument("--recontextualize", action="store_true",
                     help="resumable pass that (re-)runs LLM contextualization only on chunks not yet "
                          "contextualized (contextualized=false — e.g. rows from a prior "
                          "--skip-contextualization run), instead of skipping anything that merely "
                          "exists. Safe to Ctrl-C/kill and re-run arbitrarily. Requires contextualization "
                          "to be enabled (incompatible with --skip-contextualization) and is meaningless "
                          "combined with --force (which already reprocesses everything unconditionally).")
    args = ap.parse_args()

    if args.recontextualize and args.skip_contextualization:
        print("--recontextualize requires LLM contextualization to be enabled — "
              "it can't be combined with --skip-contextualization.")
        sys.exit(1)

    if args.recontextualize and args.force:
        print("--recontextualize and --force are redundant with each other — --force already "
              "reprocesses every chunk unconditionally, which includes recontextualizing. Pick one.")
        sys.exit(1)

    if args.wipe and (args.adventure or args.book or args.source_type):
        print(
            "--wipe clears the WHOLE collection and is meant to be combined with a full "
            "unscoped rebuild. Combining it with --adventure/--book/--source-type would wipe "
            "everything but only rebuild the scoped subset, silently leaving the rest empty. "
            "Run --wipe alone for a full rebuild, or a scope flag alone (no --wipe) for a "
            "per-book incremental rebuild."
        )
        sys.exit(1)

    if args.fresh and args.wipe:
        print("--fresh is redundant with --wipe — --wipe already clears everything unconditionally.")
        sys.exit(1)

    if args.fresh and not (args.book or args.adventure or args.source_type == "core"):
        print("--fresh has nothing to scope to — combine it with --book/--adventure/"
              "--source-type core, or use --wipe for a full unscoped rebuild.")
        sys.exit(1)

    if args.fresh and args.recontextualize:
        print("--fresh deletes rows outright; --recontextualize is purely additive/resumable. "
              "These are contradictory — pick one.")
        sys.exit(1)

    ollama_url = args.ollama_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA
    vllm_url = args.vllm_url or os.environ.get("VLLM_BASE_URL") or DEFAULT_VLLM
    source_dir = Path(args.source)

    if not source_dir.exists():
        print(f"Source dir '{source_dir}' not found. Run ocr_ingest.py first.")
        sys.exit(1)

    # --book's value matches extract_entities.py's raw-filename-stem convention;
    # book metadata (set in build_documents) is the .title()-cased display form —
    # normalize the same way here so the same --book value works on both scripts.
    book_display_name = args.book.replace("-", " ").replace("_", " ").title() if args.book else None

    engine = _get_engine()
    from backend.stores import tables as t

    # A scoped run (--book/--adventure/--source-type) is incremental by default
    # — it never deletes existing rows, only --wipe (full, unscoped) or --fresh
    # (scoped, explicit opt-in) do. (2026-07-13 incident: a scoped run used to
    # delete its scope's existing rows unconditionally, no opt-out — that run
    # happened to combine with docs/source/core not being mounted into the
    # container yet, see docker-compose.yml's comment, so build_documents()
    # found zero core docs afterward and net-deleted every core rule_chunks
    # row with nothing to replace them. --fresh makes deletion an explicit
    # choice instead of scoping's default behavior.)
    if args.wipe:
        _wipe_table(engine)
    elif args.fresh and args.book:
        _delete_where(engine, t.rule_chunks.c.book == book_display_name)
        print(f"  Removed existing chunks for book '{book_display_name}'.")
    elif args.fresh and args.adventure:
        _delete_where(engine, t.rule_chunks.c.adventure == args.adventure)
        print(f"  Removed existing chunks for '{args.adventure}'.")
    elif args.fresh and args.source_type == "core":
        _delete_where(engine, t.rule_chunks.c.source_type == "core")
        print(f"  Removed existing core chunks.")

    print(f"\nScanning {source_dir} …")
    all_docs = build_documents(source_dir)

    if args.book:
        all_docs = [d for d in all_docs if d["meta"]["book"] == book_display_name]
    elif args.adventure:
        all_docs = [d for d in all_docs if d["meta"]["adventure"] == args.adventure]
    elif args.source_type:
        all_docs = [d for d in all_docs if d["meta"]["source_type"] == args.source_type]

    if not all_docs:
        print("Nothing to index.")
        return

    # Attach each child's parent text for contextualization — computed here
    # (still in-memory from build_documents' pass) rather than re-derived
    # per chunk during indexing.
    parent_text_by_id = {d["id"]: d["text"] for d in all_docs if d["meta"]["granularity"] == "parent"}
    for d in all_docs:
        if d["meta"]["granularity"] == "child":
            d["_parent_text"] = parent_text_by_id.get(d["meta"].get("parent_chunk_id", ""), d["text"])

    print(f"\nIndexing {len(all_docs)} chunks (parent+child, resumable) …")
    _index_documents(
        all_docs, engine, ollama_url, vllm_url, args.skip_contextualization, args.force, args.context_model,
        recontextualize=args.recontextualize,
    )

    print(f"\nDone. {len(all_docs)} chunks in rule_chunks. (content_tsv keyword-search index "
          "updates automatically — no separate rebuild step.)")


if __name__ == "__main__":
    main()
