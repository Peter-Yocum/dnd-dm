#!/usr/bin/env python3
"""
backfill_history_chunks.py — re-embed every existing Campaign.sessions
chronicle into HistoryStore's new per-event chunk schema (one document per
summary paragraph + one per key event, each contextualized), replacing the
old whole-session-as-one-document embedding.

Source of truth is Postgres (Campaign.sessions, already durable) — this
script reads from there, not from the old Chroma documents, so there's no
data-loss window even if it's interrupted or re-run out of order. Idempotent:
a session already migrated (probed via its first new-schema chunk id) is
skipped. The old whole-session document (Chroma id == session.id) is deleted
after a successful migration.

Usage:
    python backfill_history_chunks.py             # apply the migration
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


def _already_migrated(history: HistoryStore, session_id: str) -> bool:
    probe_ids = [f"{session_id}::summary::0", f"{session_id}::key_event::0"]
    found = history._chroma()._collection.get(ids=probe_ids).get("ids", [])
    return bool(found)


async def main(dry_run: bool) -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)
    history = HistoryStore()

    total_migrated = 0
    total_skipped = 0

    for summary in await store.list_all():
        campaign = await store.load(summary.id)
        if not campaign or not campaign.sessions:
            continue

        for session in campaign.sessions:
            if _already_migrated(history, session.id):
                total_skipped += 1
                continue

            print(f"  [{campaign.name}] session {session.session_number} ({session.id})")
            if not dry_run:
                history.add_session(
                    campaign.id, session.id, session.session_number, session.summary, session.key_events,
                )
                try:
                    history._chroma()._collection.delete(ids=[session.id])
                except Exception:
                    pass
            total_migrated += 1

    verb = "Would migrate" if dry_run else "Migrated"
    print(f"\n{verb} {total_migrated} session(s); skipped {total_skipped} (already migrated).")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
