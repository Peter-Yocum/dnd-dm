#!/usr/bin/env python3
"""
build_index.py — Chunk, contextualize, and embed docs/source/ into ChromaDB
for hybrid (dense + BM25) RAG, with a parent/child chunk split.

Expected folder structure:
    docs/source/core/                       ← searched for every campaign
    docs/source/adventures/{slug}/          ← searched only for campaigns that include {slug}
    docs/source/adventures/{slug}/_meta.json  ← optional display metadata

Each .md file is split on ## headers into PARENT sections (as before), and
each parent is further split into smaller CHILD sub-chunks. Only children are
embedded for dense search and indexed for BM25 (see backend/rag/hybrid.py);
parents exist purely as an id-addressable lookup target so a search hit can
be expanded to its full surrounding section (backend/stores/rules_store.py's
search()). Every child chunk is contextualized before embedding — a short
LLM-generated blurb is prepended to the text that gets EMBEDDED, but the
chunk's stored/citable text (`documents` in Chroma, `.content` on RuleChunk)
stays the raw, unmodified book text.

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
chunk_id, this script checks whether that id already exists in the target
Chroma collection and skips it if so (unless --force). Chroma's own
persistent storage IS the completion tracker — no second cache file to keep
in sync. Chunks are embedded and upserted in small batches (not buffered for
a whole book), so a kill loses at most one small in-flight batch.

IMPORTANT — --wipe and resuming after a kill: --wipe clears the whole
collection ONCE, then indexing proceeds with the resumable behavior above.
If a --wipe run is interrupted, resume with a PLAIN run (no --wipe) — NOT by
repeating --wipe, which would destroy the progress already saved. See `make
index` vs `make reindex-full` in the Makefile.

Usage:
    python build_index.py                        # index everything (resumable)
    python build_index.py --wipe                  # FIRST run only: clear collection, then index
    python build_index.py --source-type core      # core books only
    python build_index.py --adventure tyranny-of-dragons  # one adventure only
    python build_index.py --skip-contextualization  # fast dev path, no LLM calls
    python build_index.py --force                 # re-process even already-indexed chunk_ids

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
DEFAULT_CHROMA  = "data/chroma_db"
DEFAULT_OLLAMA  = "http://localhost:11434"
DEFAULT_EMBED   = "nomic-embed-text"
COLLECTION      = "rules"
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
            # chunk_id is duplicated into metadata (not just used as the
            # Chroma document id) because langchain's similarity_search()
            # returns Document objects with no id field — metadata is the
            # only thing callers can read back, and Stage 2's citation
            # checks need a real, round-trippable chunk_id.
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


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _get_chroma(chroma_dir: str, ollama_url: str):
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OllamaEmbeddings(model=DEFAULT_EMBED, base_url=ollama_url),
        persist_directory=chroma_dir,
    )


def _wipe_collection(chroma_dir: str) -> None:
    import chromadb
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION)
        print(f"  Wiped '{COLLECTION}' collection.")
    except Exception:
        pass


def _delete_where(chroma_dir: str, where: dict) -> None:
    import chromadb
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.get_collection(COLLECTION).delete(where=where)
    except Exception:
        pass


_BM25_FETCH_PAGE_SIZE = 5_000  # chromadb's underlying SQLite store errors with
# "too many SQL variables" on an unpaginated get() once a collection grows large
# enough (confirmed live once the corpus passed ~250k+ child chunks across
# several reindexed books) — page through with limit/offset instead of one
# unbounded call.


def _rebuild_bm25(chroma_dir: str, ollama_url: str) -> None:
    """Always rebuilt from the CURRENT full Chroma state, regardless of this
    run's scope (--wipe/--adventure/--source-type) — cheap, pure CPU, avoids
    a stale-BM25 bug class where a partial reindex forgets to refresh it.
    Paginated (see _BM25_FETCH_PAGE_SIZE) since an unbounded get() over the
    whole collection hits a real SQLite bound-variable limit once the corpus
    is large."""
    from backend.rag.hybrid import BM25Index

    store = _get_chroma(chroma_dir, ollama_url)
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    offset = 0
    while True:
        page = store._collection.get(
            where={"granularity": {"$eq": "child"}},
            limit=_BM25_FETCH_PAGE_SIZE, offset=offset,
            include=["documents", "metadatas"],
        )
        page_ids = page.get("ids", [])
        if not page_ids:
            break
        ids.extend(page_ids)
        texts.extend(page.get("documents", []))
        metadatas.extend(page.get("metadatas", []))
        offset += len(page_ids)
        if len(page_ids) < _BM25_FETCH_PAGE_SIZE:
            break

    bm25 = BM25Index.build(ids, texts, metadatas)
    out_path = str(Path(chroma_dir).parent / "bm25_rules.pkl")
    bm25.save(out_path)
    print(f"  Rebuilt BM25 index — {len(ids)} child chunks — {out_path}")


# ── indexing (resumable) ──────────────────────────────────────────────────────

def _index_documents(
    all_docs: list[dict], chroma_dir: str, ollama_url: str,
    skip_contextualization: bool, force: bool, context_model: str | None = None,
) -> None:
    from langchain_ollama import OllamaEmbeddings
    from tqdm import tqdm
    from backend.rag.contextualizer import ChunkContextualizer

    store = _get_chroma(chroma_dir, ollama_url)
    collection = store._collection
    embeddings_fn = OllamaEmbeddings(model=DEFAULT_EMBED, base_url=ollama_url)
    if skip_contextualization:
        contextualizer = None
    elif context_model:
        contextualizer = ChunkContextualizer(model=context_model, ollama_base_url=ollama_url)
    else:
        contextualizer = ChunkContextualizer(ollama_base_url=ollama_url)

    total = len(all_docs)
    indexed_count = 0
    skipped_count = 0
    progress = tqdm(total=total, unit="chunk", desc="Indexing", dynamic_ncols=True)
    for i in range(0, total, INDEX_BATCH_SIZE):
        batch = all_docs[i:i + INDEX_BATCH_SIZE]
        batch_ids = [d["id"] for d in batch]

        if not force:
            existing = set(collection.get(ids=batch_ids).get("ids", []))
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

            vectors = embeddings_fn.embed_documents(embed_texts)
            collection.upsert(
                ids=[d["id"] for d in batch],
                embeddings=vectors,
                documents=[d["text"] for d in batch],
                metadatas=[d["meta"] for d in batch],
            )

        batch_size = len(batch_ids)
        indexed_count += len(batch)
        skipped_count += batch_size - len(batch)
        progress.set_postfix(indexed=indexed_count, skipped=skipped_count)
        progress.update(batch_size)
    progress.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the hybrid ChromaDB index from docs/source/.")
    ap.add_argument("--source",      default=DEFAULT_SOURCE)
    ap.add_argument("--chroma",      default=DEFAULT_CHROMA)
    ap.add_argument("--ollama-url",  default=None)
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
    ap.add_argument("--skip-contextualization", action="store_true",
                     help="fast dev path: skip the LLM contextualization pass entirely")
    ap.add_argument("--context-model", default=None,
                     help="Ollama model for the contextualization pass (default: settings.mechanics_model, "
                          "gemma4:26b-mlx — Apple Silicon MLX format, ~26B, won't fit a 12GB-class GPU or "
                          "load at all on non-Apple hardware). On a machine that can't run the default, pass "
                          "a smaller model here explicitly rather than silently falling back to it — this "
                          "codebase has a documented incident (design.md) where an unvalidated smaller model "
                          "(qwen2.5:14b) produced fake tool calls/garbled output under sustained use, so "
                          "prefer a model from an already-trusted family (e.g. gemma4:e4b, the desktop's "
                          "target model — see docs/engineering-notes/desktop-native-ingestion.md) over an "
                          "unvalidated one, and smoke-test before trusting it for a long unattended run.")
    ap.add_argument("--force",       action="store_true",
                     help="re-process chunk_ids that already exist in the collection")
    args = ap.parse_args()

    if args.wipe and (args.adventure or args.book or args.source_type):
        print(
            "--wipe clears the WHOLE collection and is meant to be combined with a full "
            "unscoped rebuild. Combining it with --adventure/--book/--source-type would wipe "
            "everything but only rebuild the scoped subset, silently leaving the rest empty. "
            "Run --wipe alone for a full rebuild, or a scope flag alone (no --wipe) for a "
            "per-book incremental rebuild."
        )
        sys.exit(1)

    ollama_url = args.ollama_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA
    source_dir = Path(args.source)

    if not source_dir.exists():
        print(f"Source dir '{source_dir}' not found. Run ocr_ingest.py first.")
        sys.exit(1)

    # --book's value matches extract_entities.py's raw-filename-stem convention;
    # book metadata (set in build_documents) is the .title()-cased display form —
    # normalize the same way here so the same --book value works on both scripts.
    book_display_name = args.book.replace("-", " ").replace("_", " ").title() if args.book else None

    if args.wipe:
        _wipe_collection(args.chroma)
    elif args.book:
        _delete_where(args.chroma, {"book": {"$eq": book_display_name}})
        print(f"  Removed existing chunks for book '{book_display_name}'.")
    elif args.adventure:
        _delete_where(args.chroma, {"adventure": {"$eq": args.adventure}})
        print(f"  Removed existing chunks for '{args.adventure}'.")
    elif args.source_type == "core":
        _delete_where(args.chroma, {"source_type": {"$eq": "core"}})
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
    _index_documents(all_docs, args.chroma, ollama_url, args.skip_contextualization, args.force, args.context_model)

    _rebuild_bm25(args.chroma, ollama_url)

    print(f"\nDone. {len(all_docs)} chunks in {args.chroma}/{COLLECTION}.")


if __name__ == "__main__":
    main()
