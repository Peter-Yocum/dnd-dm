#!/usr/bin/env python3
"""
build_index.py — Chunk and embed docs/source/ into ChromaDB for RAG.

Expected folder structure:
    docs/source/core/                       ← searched for every campaign
    docs/source/adventures/{slug}/          ← searched only for campaigns that include {slug}
    docs/source/adventures/{slug}/_meta.json  ← optional display metadata

Each .md file is split on ## headers into sections, embedded with
nomic-embed-text via Ollama, and stored in ChromaDB with metadata:
    book        — filename stem formatted as title
    section     — ## header text
    source_type — "core" | "adventure"
    adventure   — slug (empty string for core)

Usage:
    python build_index.py                        # index everything
    python build_index.py --wipe                 # clear collection, then index
    python build_index.py --source-type core     # core books only
    python build_index.py --adventure tyranny-of-dragons  # one adventure only

Run inside the app container via:  make index
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_SOURCE  = "docs/source"
DEFAULT_CHROMA  = "data/chroma_db"
DEFAULT_OLLAMA  = "http://localhost:11434"
DEFAULT_EMBED   = "nomic-embed-text"
COLLECTION      = "rules"
MAX_CHUNK_CHARS = 1500
BATCH_SIZE      = 64    # documents per ChromaDB add call

# ── chunking ──────────────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r'^#{2,3}\s+(.+)$', re.MULTILINE)


def chunk_markdown(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str, str]]:
    """Split markdown into (section_title, chunk_text) pairs on ## / ### headers."""
    matches = list(_HEADER_RE.finditer(text))
    sections: list[tuple[str, str]] = []

    def _add(title: str, body: str) -> None:
        body = body.strip()
        if not body:
            return
        if len(body) <= max_chars:
            sections.append((title, f"## {title}\n\n{body}"))
        else:
            words = body.split()
            chunk, length, part = [], 0, 1
            for word in words:
                if length + len(word) + 1 > max_chars and chunk:
                    sections.append((f"{title} ({part})", f"## {title}\n\n{' '.join(chunk)}"))
                    chunk, length, part = [word], len(word), part + 1
                else:
                    chunk.append(word)
                    length += len(word) + 1
            if chunk:
                label = f"{title} ({part})" if part > 1 else title
                sections.append((label, f"## {label}\n\n{' '.join(chunk)}"))

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

def _chunk_id(source_type: str, adventure: str, book: str, section: str, sample: str, ordinal: int) -> str:
    """ordinal is the chunk's position within its own book (reset per file) —
    content hashing alone isn't guaranteed unique: two distinct chunks can
    share the same section title and leading text (e.g. a generic heading
    like "Item" appearing more than once in a table with thin bodies),
    which without ordinal collides into the same id and makes ChromaDB
    reject the whole upsert batch with a DuplicateIDError."""
    key = f"{source_type}::{adventure}::{book}::{section}::{sample}::{ordinal}"
    return hashlib.md5(key.encode()).hexdigest()


def build_documents(source_dir: Path) -> list[dict]:
    """Walk source_dir and return a list of {id, text, metadata} dicts."""
    docs: list[dict] = []
    core_dir = source_dir / "core"
    adv_dir  = source_dir / "adventures"

    if core_dir.exists():
        before = len(docs)
        for md_file in sorted(core_dir.glob("*.md")):
            book_name = md_file.stem.replace("-", " ").replace("_", " ").title()
            text = md_file.read_text(encoding="utf-8")
            for ordinal, (section, chunk_text) in enumerate(chunk_markdown(text)):
                docs.append({
                    "id":   _chunk_id("core", "", book_name, section, chunk_text[:32], ordinal),
                    "text": chunk_text,
                    "meta": {"book": book_name, "section": section,
                             "source_type": "core", "adventure": ""},
                })
        print(f"  core/  — {len(docs) - before} chunks")

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
                text = md_file.read_text(encoding="utf-8")
                for ordinal, (section, chunk_text) in enumerate(chunk_markdown(text)):
                    docs.append({
                        "id":   _chunk_id("adventure", slug, book_name, section, chunk_text[:32], ordinal),
                        "text": chunk_text,
                        "meta": {"book": book_name, "section": section,
                                 "source_type": "adventure", "adventure": slug},
                    })
            print(f"  adventures/{slug}/  ({adv_name}) — {len(docs) - before} chunks")

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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build ChromaDB index from docs/source/.")
    ap.add_argument("--source",      default=DEFAULT_SOURCE)
    ap.add_argument("--chroma",      default=DEFAULT_CHROMA)
    ap.add_argument("--ollama-url",  default=None)
    ap.add_argument("--wipe",        action="store_true", help="clear whole collection first")
    ap.add_argument("--source-type", choices=["core", "adventure"], default=None)
    ap.add_argument("--adventure",   default=None, help="index one adventure slug only")
    args = ap.parse_args()

    ollama_url = args.ollama_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA
    source_dir = Path(args.source)

    if not source_dir.exists():
        print(f"Source dir '{source_dir}' not found. Run ocr_ingest.py first.")
        sys.exit(1)

    if args.wipe:
        _wipe_collection(args.chroma)
    elif args.adventure:
        _delete_where(args.chroma, {"adventure": {"$eq": args.adventure}})
        print(f"  Removed existing chunks for '{args.adventure}'.")
    elif args.source_type == "core":
        _delete_where(args.chroma, {"source_type": {"$eq": "core"}})
        print(f"  Removed existing core chunks.")

    print(f"\nScanning {source_dir} …")
    all_docs = build_documents(source_dir)

    if args.adventure:
        all_docs = [d for d in all_docs if d["meta"]["adventure"] == args.adventure]
    elif args.source_type:
        all_docs = [d for d in all_docs if d["meta"]["source_type"] == args.source_type]

    if not all_docs:
        print("Nothing to index.")
        return

    print(f"\nIndexing {len(all_docs)} chunks …")
    store = _get_chroma(args.chroma, ollama_url)

    from langchain_core.documents import Document
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch   = all_docs[i: i + BATCH_SIZE]
        store.add_documents(
            [Document(page_content=d["text"], metadata=d["meta"]) for d in batch],
            ids=[d["id"] for d in batch],
        )
        print(f"  {min(i + BATCH_SIZE, len(all_docs))}/{len(all_docs)}")

    print(f"\nDone. {len(all_docs)} chunks in {args.chroma}/{COLLECTION}.")


if __name__ == "__main__":
    main()
