from langchain_core.tools import BaseTool

from backend.stores.campaign_store import CampaignStore
from backend.stores.history_store import HistoryStore
from backend.stores.rules_store import RulesStore
from backend.tools import (
    campaign, chargen, combat, companion, dice, levelup, memory, npc, party, quest, resolution, rules, world,
)


def get_tools(
    campaign_id: str,
    store: CampaignStore,
    rules_store: RulesStore,
    history_store: HistoryStore,
    books_in_play: list[str],
) -> list[BaseTool]:
    """Assemble the full DM tool set bound to a specific campaign session."""
    return [
        *dice.make_tools(),
        *rules.make_tools(rules_store, books_in_play),
        *memory.make_tools(history_store, campaign_id),
        *party.make_tools(campaign_id, store),
        *npc.make_tools(campaign_id, store),
        *combat.make_tools(campaign_id, store),
        *resolution.make_tools(campaign_id, store),
        *world.make_tools(campaign_id, store),
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
) -> list[BaseTool]:
    """Restricted tool set for the automatic world-prep background pass:
    read adventure text (search_rules) and author region-scale locations/
    connections. No party, combat, quest, movement, or travel tools."""
    return [
        *rules.make_tools(rules_store, books_in_play),
        *world.make_authoring_tools(campaign_id, store),
    ]
