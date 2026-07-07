#!/usr/bin/env python3
"""
merge_chroma.py — Merge a second machine's ChromaDB (e.g. a desktop doing
bulk OCR/indexing) into this machine's canonical rules collection, then
rebuild the BM25 pickle from the merged result.

Safe by construction, not by convention: every chunk_id in this collection
is a deterministic MD5 hash of its content (source_type::adventure::book::
section::sample::ordinal::part — see build_index.py's _chunk_id()), not a
random UUID. So:
  - Two machines that processed DIFFERENT books produce disjoint id sets —
    merging is a pure union, zero collision risk.
  - If the same book was somehow processed on both machines, the resulting
    ids are identical too — merging just upserts one copy over the other
    (whichever is merged last "wins"), never creates a duplicate or a
    corrupt mixed record.
Re-running this script against the same source is always safe (idempotent
upsert), e.g. if you sync the desktop's chroma_db again after it's done more
books overnight.

Usage:
    python scripts/merge_chroma.py --source /path/to/desktop/chroma_db
    python scripts/merge_chroma.py --source /path/to/desktop/chroma_db --target data/chroma_db
"""

import argparse
import sys
from pathlib import Path

# Windows' default console codec (cp1252/"charmap") can't encode the em-dashes
# this script prints for readability — see ocr_ingest.py's identical fix for
# the confirmed-live crash this avoids.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings

DEFAULT_TARGET = "data/chroma_db"
COLLECTION = "rules"
# Deliberately smaller than build_index.py's 5,000-chunk _BM25_FETCH_PAGE_SIZE
# bound (which exists to dodge a SQLite bound-variable limit, not a memory
# limit). This script runs via `docker compose exec` INTO the same
# container/cgroup as the live `uvicorn --reload` app process — confirmed
# live that a 5,000-chunk page (with full embedding vectors, on top of the
# already-running server) OOM-killed the container (`docker inspect`
# .State.OOMKilled == true) even though this whole Docker Desktop VM has
# under 1GB total, shared with the db container too. 500 leaves enough
# headroom for the server that's still running.
_PAGE_SIZE = 500


def _get_chroma(chroma_dir: str, ollama_url: str, collection: str):
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    return Chroma(
        collection_name=collection,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url),
        persist_directory=chroma_dir,
    )


def merge(source_dir: str, target_dir: str, ollama_url: str, collection: str) -> int:
    """Streams one page at a time from source straight into target — never
    holds more than one page's embeddings/documents/metadatas in memory at
    once. Buffering the WHOLE corpus before writing anything (the first cut
    of this function) OOM-killed under this container's tight memory limit
    (confirmed live: exit 137, the same failure class as the earlier
    CrossEncoderReranker incident) — the container is small on purpose (see
    reranker.py's docstring), so any bulk pass here has to respect that."""
    if not Path(source_dir).exists():
        print(f"Source not found: {source_dir}", file=sys.stderr)
        sys.exit(1)
    if Path(source_dir).resolve() == Path(target_dir).resolve():
        print("Source and target are the same directory — nothing to merge.", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming '{collection}' from source into target ({_PAGE_SIZE}-chunk pages)")
    print(f"  source: {source_dir}")
    print(f"  target: {target_dir}")
    source = _get_chroma(source_dir, ollama_url, collection)
    target = _get_chroma(target_dir, ollama_url, collection)

    total = 0
    offset = 0
    while True:
        page = source._collection.get(
            limit=_PAGE_SIZE, offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        page_ids = page.get("ids", [])
        if not page_ids:
            break
        target._collection.upsert(
            ids=page_ids,
            embeddings=page.get("embeddings", []),
            documents=page.get("documents", []),
            metadatas=page.get("metadatas", []),
        )
        total += len(page_ids)
        print(f"  merged {total} chunk(s) so far...")
        offset += len(page_ids)
        if len(page_ids) < _PAGE_SIZE:
            break

    print(f"  Merged {total} chunk(s) into target.")
    return total


def _rebuild_bm25(target_dir: str, ollama_url: str, collection: str) -> None:
    """Same rebuild-from-current-Chroma-state approach build_index.py's own
    _rebuild_bm25 uses. Unlike merge()'s embeddings, BM25 only needs raw
    text + metadata — the same corpus size that build_index.py/
    history_store.py already rebuild this way successfully in this
    container, so paginated accumulation (not page-by-page streaming) is
    fine here; BM25Okapi needs the whole tokenized corpus in memory to build
    its index regardless, there's no incremental construction to stream
    into."""
    from backend.rag.hybrid import BM25Index

    target = _get_chroma(target_dir, ollama_url, collection)
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    offset = 0
    while True:
        page = target._collection.get(
            limit=_PAGE_SIZE, offset=offset, include=["documents", "metadatas"],
        )
        page_ids = page.get("ids", [])
        if not page_ids:
            break
        ids.extend(page_ids)
        documents.extend(page.get("documents", []))
        metadatas.extend(page.get("metadatas", []))
        offset += len(page_ids)
        if len(page_ids) < _PAGE_SIZE:
            break

    bm25 = BM25Index.build(ids, documents, metadatas)
    out_path = str(Path(target_dir).parent / f"bm25_{'rules' if collection == 'rules' else collection}.pkl")
    bm25.save(out_path)
    print(f"  Rebuilt BM25 index — {len(ids)} chunk(s) — {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="path to the other machine's chroma_db directory")
    ap.add_argument("--target", default=DEFAULT_TARGET, help=f"path to this machine's canonical chroma_db (default: {DEFAULT_TARGET})")
    ap.add_argument("--collection", default=COLLECTION, help=f"Chroma collection name to merge (default: {COLLECTION})")
    ap.add_argument("--ollama-url", default=settings.ollama_base_url)
    ap.add_argument("--skip-bm25-rebuild", action="store_true", help="merge Chroma only, don't rebuild the BM25 pickle")
    args = ap.parse_args()

    merged = merge(args.source, args.target, args.ollama_url, args.collection)
    if merged and not args.skip_bm25_rebuild:
        _rebuild_bm25(args.target, args.ollama_url, args.collection)
    print("Done.")


if __name__ == "__main__":
    main()
