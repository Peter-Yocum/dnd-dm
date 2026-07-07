#!/usr/bin/env python3
"""
load_lore_json.py — Load extract_entities.py's JSON registries into Postgres
(lore_entities/lore_entity_aliases) without re-running any LLM extraction.

Exists for the desktop/laptop split: the desktop does bulk OCR + indexing +
entity extraction natively (no Docker — no virtualization available there),
which also means no Postgres running on it. extract_entities.py is run there
WITHOUT --write-postgres, so it produces only the JSON registry (its normal
debug/audit artifact — see extract_entities.py's docstring). Once that JSON
is synced back to this machine (which owns the canonical Postgres instance),
this script performs the exact same upsert extract_entities.py would have
done directly, reusing LoreStore.upsert_entity so there's exactly one place
that upsert logic lives.

Safe to re-run: LoreStore.upsert_entity is keyed on
(book_slug, entity_type, canonical_name), so loading the same registry twice
just re-upserts the same rows, never duplicates them.

Usage:
    python scripts/load_lore_json.py --book "D&D 5.5E - Player's Handbook" --source-type core
    python scripts/load_lore_json.py --book "Curse of Strahd"                 # adventure (default)
    python scripts/load_lore_json.py --all-core
    python scripts/load_lore_json.py --all-adventures
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.stores.lore_store import LoreStore

CORE_ENTITIES_DIR = Path("docs/source/core/_entities")
ADVENTURES_DIR = Path("docs/source/adventures")


def _iter_registries(args: argparse.Namespace) -> list[tuple[Path, str, str]]:
    """Returns a list of (json_path, book_slug, source_type) to load."""
    results: list[tuple[Path, str, str]] = []
    if args.book:
        if args.source_type == "core":
            results.append((CORE_ENTITIES_DIR / f"{args.book}.json", args.book, "core"))
        else:
            results.append((ADVENTURES_DIR / args.book / "_entities.json", args.book, "adventure"))
    if args.all_core:
        results.extend((p, p.stem, "core") for p in sorted(CORE_ENTITIES_DIR.glob("*.json")))
    if args.all_adventures:
        results.extend(
            (p, p.parent.name, "adventure")
            for p in sorted(ADVENTURES_DIR.glob("*/_entities.json"))
        )
    return results


async def _load_one(
    lore_store: LoreStore, path: Path, book: str, source_type: str, force_incomplete: bool,
) -> int:
    if not path.exists():
        print(f"  SKIP {book}: {path} not found")
        return 0
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not registry.get("_complete") and not force_incomplete:
        print(f"  SKIP {book}: registry not marked _complete — an interrupted run on the "
              f"source machine (use --force-incomplete to load the entities finished so far anyway)")
        return 0

    count = 0
    for name, entry in registry.items():
        if name.startswith("_"):  # "_complete", "_connections" — not an entity
            continue
        profile = entry["profile"]
        await lore_store.upsert_entity(
            book_slug=book,
            entity_type=entry["type"],
            canonical_name=name,
            profile=profile,
            aliases=entry.get("aliases", []),
            spoiler_tier=profile.get("spoiler_tier", "public"),
            source_type=source_type,
        )
        count += 1
    print(f"  {book}: loaded {count} entit(y/ies) from {path}")
    return count


async def _main_async() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default=None)
    ap.add_argument("--source-type", choices=["core", "adventure"], default="adventure")
    ap.add_argument("--all-core", action="store_true", help="load every docs/source/core/_entities/*.json")
    ap.add_argument("--all-adventures", action="store_true", help="load every docs/source/adventures/*/_entities.json")
    ap.add_argument("--force-incomplete", action="store_true",
                    help="load a registry not marked _complete (e.g. syncing mid-run from the desktop)")
    args = ap.parse_args()

    if not (args.book or args.all_core or args.all_adventures):
        print("Nothing to do — pass --book, --all-core, or --all-adventures", file=sys.stderr)
        sys.exit(1)

    registries = _iter_registries(args)
    if not registries:
        print("No matching registries found.", file=sys.stderr)
        sys.exit(1)

    lore_store = LoreStore(create_async_engine(settings.database_url))
    total = 0
    for path, book, source_type in registries:
        total += await _load_one(lore_store, path, book, source_type, args.force_incomplete)
    print(f"Done — {total} entit(y/ies) loaded across {len(registries)} book(s).")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
