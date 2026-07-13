#!/usr/bin/env python3
"""
backfill_session_progress.py — re-summarize sessions recorded before
Session.adventure_progress existed (and before summarize_session cross-checked
its chronicle against real party state / retrieved adventure text).

Session.adventure_progress (backend/models.py) is a new field — any session
saved before it shipped has an empty string there, which is harmless at
runtime (build_session_kickoff_message just skips the extra re-grounding step
for that one transition) but means:
  - The next session's kickoff never gets a fresh search_rules lookup keyed
    on where the party actually is in the module.
  - The old chronicle/key_events were written without the party's real
    inventory/currency as ground truth, so a narrated-but-never-applied claim
    (e.g. "stole their weapons" when no add_item_to_character ever backed it)
    can be baked into permanent campaign history.

This re-runs summarize_session() against each affected session's original
transcript (via its stored thread_id) and overwrites summary/key_events/
adventure_progress with the regenerated, grounded versions.

Untouched check is an empty adventure_progress — any session that already has
one (e.g. a previous run of this script) is left alone. Safe to run more than
once; idempotent.

Usage:
    python backfill_session_progress.py                        # all campaigns
    python backfill_session_progress.py --campaign-id <id>      # one campaign
    python backfill_session_progress.py --dry-run               # report only

Run inside the app container via:
    docker compose exec app python scripts/backfill_session_progress.py
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running as `python scripts/backfill_session_progress.py` from
# anywhere — Python sets sys.path[0] to this script's own directory, not the
# repo root, so `backend` wouldn't otherwise be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.agent.dm_agent import agent_lifespan, summarize_session
from backend.config import settings
from backend.stores.campaign_store import CampaignStore
from backend.stores.rules_store import RulesStore


def _untouched(session) -> bool:
    return not session.adventure_progress


async def main(dry_run: bool, campaign_id: str | None) -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)
    rules_store = RulesStore(engine, settings.vllm_embed_base_url)

    total_backfilled = 0
    total_skipped_touched = 0
    total_skipped_no_thread = 0
    total_skipped_empty_thread = 0

    # summarize_session() reads transcripts via get_thread_messages(), which
    # reads from dm_agent's module-level checkpointer — only ever initialized
    # inside this lifespan context manager (normally by FastAPI's own
    # startup). Without it, get_thread_messages silently returns [] for every
    # thread, which looks exactly like "no dialogue to summarize" instead of
    # the real problem — do not remove this wrapper.
    async with agent_lifespan():
        campaign_ids = [campaign_id] if campaign_id else [s.id for s in await store.list_all()]

        for cid in campaign_ids:
            campaign = await store.load(cid)
            if not campaign or not campaign.sessions:
                continue

            changed = False
            for session in campaign.sessions:
                if not _untouched(session):
                    total_skipped_touched += 1
                    continue
                if not session.thread_id:
                    print(f"  [{campaign.name}] session {session.session_number}: no thread_id, skipped")
                    total_skipped_no_thread += 1
                    continue

                print(f"  [{campaign.name}] session {session.session_number}: re-summarizing (thread {session.thread_id})...")
                summary, key_events, adventure_progress = await summarize_session(
                    session.thread_id, campaign, rules_store
                )
                if summary == "Session contained no dialogue to summarize.":
                    print(f"    no checkpoint history found for this thread — skipped")
                    total_skipped_empty_thread += 1
                    continue
                print(f"    adventure_progress: {adventure_progress!r}")

                if not dry_run:
                    session.summary = summary
                    session.key_events = key_events
                    session.adventure_progress = adventure_progress
                    changed = True
                total_backfilled += 1

            if changed:
                await store.save(campaign)

    verb = "Would backfill" if dry_run else "Backfilled"
    print(
        f"\n{verb} {total_backfilled} session(s); "
        f"skipped {total_skipped_touched} already-touched, {total_skipped_no_thread} with no thread_id, "
        f"{total_skipped_empty_thread} with no retrievable transcript."
    )
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    parser.add_argument("--campaign-id", default=None, help="Only backfill this campaign.")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run, args.campaign_id))
