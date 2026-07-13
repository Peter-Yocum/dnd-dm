#!/usr/bin/env python3
"""
backfill_history_chunks.py — (re-)embed every existing Campaign.sessions
chronicle into session_chronicle_chunks (one document per summary paragraph
+ one per key event, each contextualized).

Source of truth is Postgres (Campaign.sessions, already durable) — this
script reads from there, so there's no data-loss window even if it's
interrupted or re-run out of order. Idempotent: add_session() upserts on
chunk_id (f"{session_id}::{event_type}::{i}"), so re-running this for a
session that's already been embedded just overwrites the same rows with
freshly-computed embeddings, not a duplicate.

2026-07-12: this also serves as the regeneration path for the ChromaDB ->
pgvector migration (session_chronicle_chunks replaces the old
"session_chronicles" Chroma collection) — every session's chronicle is
fully regenerable from Campaign.sessions, so there was never a need to
migrate raw Chroma data directly.

Usage:
    python backfill_history_chunks.py             # apply, all campaigns
    python backfill_history_chunks.py --campaign-id <id>
    python backfill_history_chunks.py --dry-run    # report only, no writes

Run inside the app container via:  make backfill-history-chunks
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.stores.campaign_store import CampaignStore
from backend.stores.history_store import HistoryStore


async def main(dry_run: bool, campaign_id: str | None) -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)
    history = HistoryStore(engine, settings.vllm_embed_base_url)

    total = 0
    campaign_ids = [campaign_id] if campaign_id else [s.id for s in await store.list_all()]

    for cid in campaign_ids:
        campaign = await store.load(cid)
        if not campaign or not campaign.sessions:
            continue

        for session in campaign.sessions:
            print(f"  [{campaign.name}] session {session.session_number} ({session.id})")
            if not dry_run:
                await history.add_session(
                    campaign.id, session.id, session.session_number, session.summary, session.key_events,
                )
            total += 1

    verb = "Would embed" if dry_run else "Embedded"
    print(f"\n{verb} {total} session(s).")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    parser.add_argument("--campaign-id", default=None, help="Limit to one campaign.")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run, args.campaign_id))
