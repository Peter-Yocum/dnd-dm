"""Automatic world-prep — a one-shot background pass that reads a campaign's
selected adventure text and pre-populates region-scale locations/connections
before play begins. Fired from backend/main.py as an asyncio background task;
never blocks the HTTP response that triggers it.
"""

import logging

from langchain_core.messages import HumanMessage

from backend.agent.world_prep_prompt import get_world_prep_prompt
from backend.models import WorldPrepStatus
from backend.stores.campaign_store import CampaignStore
from backend.stores.rules_store import RulesStore

log = logging.getLogger(__name__)

_SEED_QUERIES = [
    "regional overview and map",
    "travel distances and days between locations",
    "major settlements and landmarks",
]


async def run_world_prep(campaign_id: str, store: CampaignStore, rules_store: RulesStore) -> None:
    """Process each of the campaign's books_in_play one at a time, seeding
    region-scale locations/connections via a bounded, non-interactive agent
    run. Failures are caught and recorded on the campaign rather than
    propagating — this runs as an orphaned asyncio.Task with nothing awaiting
    it, so an uncaught exception here would otherwise just vanish silently.
    """
    from backend.agent.dm_agent import get_world_prep_agent  # local import: avoids a cycle with dm_agent's own imports

    campaign = await store.load(campaign_id)
    if campaign is None or not campaign.books_in_play:
        return

    campaign.world_prep_status = WorldPrepStatus.IN_PROGRESS
    await store.save(campaign)

    try:
        for book in campaign.books_in_play:
            seed_chunks = []
            for query in _SEED_QUERIES:
                seed_chunks += rules_store.search_adventure_only(query, adventure=book, k=4)
            seed_context = "\n\n---\n\n".join(
                f"[{c.book} — {c.section}]\n{c.content}" for c in seed_chunks
            )

            # Re-load: an earlier book's create_location/connect_locations
            # calls have already saved, so the agent sees prior progress.
            campaign = await store.load(campaign_id)
            agent = get_world_prep_agent(campaign, store, rules_store, books_in_play=[book])
            prompt = get_world_prep_prompt(campaign, book, seed_context)
            await agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": 60},
            )

        campaign = await store.load(campaign_id)
        campaign.world_prep_status = WorldPrepStatus.COMPLETE
        await store.save(campaign)
    except Exception as e:
        log.exception("World-prep failed for campaign %s", campaign_id)
        campaign = await store.load(campaign_id)
        if campaign is not None:
            campaign.world_prep_status = WorldPrepStatus.FAILED
            campaign.world_prep_error = str(e)[:2000]
            await store.save(campaign)
