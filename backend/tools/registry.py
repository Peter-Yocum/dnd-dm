from langchain_core.tools import BaseTool

from backend.stores.campaign_store import CampaignStore
from backend.stores.graph_store import RelationGraphStore
from backend.stores.history_store import HistoryStore
from backend.stores.lore_store import LoreStore
from backend.stores.rules_store import RulesStore
from backend.tools import (
    campaign, chargen, combat, companion, dice, levelup, lore, memory, npc, party, quest, resolution, rules, world,
)


def get_tools(
    campaign_id: str,
    store: CampaignStore,
    rules_store: RulesStore,
    history_store: HistoryStore,
    books_in_play: list[str],
    lore_store: LoreStore,
    graph_store: RelationGraphStore,
) -> list[BaseTool]:
    """Assemble the full DM tool set bound to a specific campaign session."""
    return [
        *dice.make_tools(),
        *rules.make_tools(rules_store, books_in_play),
        *lore.make_tools(campaign_id, store, lore_store, rules_store, books_in_play, graph_store),
        *memory.make_tools(history_store, campaign_id),
        *party.make_tools(campaign_id, store, lore_store, books_in_play, graph_store),
        *npc.make_tools(campaign_id, store, lore_store, books_in_play, graph_store),
        *combat.make_tools(campaign_id, store, lore_store, books_in_play),
        *resolution.make_tools(campaign_id, store),
        *world.make_tools(campaign_id, store, lore_store, books_in_play),
        *quest.make_tools(campaign_id, store),
        *campaign.make_tools(campaign_id, store),
        *companion.make_tools(campaign_id, store),
        *levelup.make_tools(campaign_id, store),
        # list_options/get_option_details — prompts.py already instructs the
        # mechanics model to call list_options('spells <class>') during
        # level-up spell selection; this tool was never bound in-game before
        # (see make_reference_tools' docstring), a structurally-impossible
        # instruction likely producing the same wrong-spell-list bug found in
        # Session 0. chargen's draft-mutating tools (update_character_draft,
        # finalize_character, etc.) stay Session-0-only via chargen.make_tools().
        *chargen.make_reference_tools(),
    ]


def get_world_prep_tools(
    campaign_id: str,
    store: CampaignStore,
    rules_store: RulesStore,
    books_in_play: list[str],
    lore_store: LoreStore,
) -> list[BaseTool]:
    """Restricted tool set for the automatic world-prep background pass:
    read adventure text (search_rules) and author region-scale locations/
    connections. No party, combat, quest, movement, or travel tools."""
    return [
        *rules.make_tools(rules_store, books_in_play),
        *lore.make_tools(campaign_id, store, lore_store, rules_store, books_in_play),
        *world.make_authoring_tools(campaign_id, store, lore_store, books_in_play),
    ]


def get_npc_prep_tools(
    campaign_id: str,
    store: CampaignStore,
    rules_store: RulesStore,
    books_in_play: list[str],
    lore_store: LoreStore,
) -> list[BaseTool]:
    """Restricted tool set for the one-shot opening-scene NPC/site-detail
    seeding pass (backend/agent/world_prep.py's second phase): read adventure
    text, create NPC records, and set the opening location's site detail. No
    party, combat, quest, movement/travel tools, and no NPC runtime tools
    (update_npc_attitude/reveal_npc_knowledge/etc.) — there's no ongoing
    conversation/attitude state to react to yet, only creation."""
    return [
        *rules.make_tools(rules_store, books_in_play),
        *lore.make_tools(campaign_id, store, lore_store, rules_store, books_in_play),
        *npc.make_authoring_tools(campaign_id, store, lore_store, books_in_play),
        *world.make_opening_detail_tools(campaign_id, store),
    ]
