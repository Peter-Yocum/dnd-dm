"""BDD-style coverage for freeform narrative time advancement + in-fiction
rests (design.md item #14) — the new advance_time/take_rest tools
(backend/tools/world.py) and the _detect_missing_time_advance_followup
guardrail (backend/agent/dm_agent.py) that backstops the Time passage
prompt section.
"""
import pytest

from backend.agent.dm_agent import _detect_missing_time_advance_followup
from backend.models import TimeOfDay
from backend.tools.world import make_travel_tools

from tests.conftest import make_character


@pytest.mark.asyncio
async def test_advance_time_advances_the_clock(store, campaign):
    campaign.time_of_day = TimeOfDay.MORNING
    campaign.days_elapsed = 5
    await store.save(campaign)

    tools = {t.name: t for t in make_travel_tools(campaign.id, store)}
    result = await tools["advance_time"].ainvoke({"hours": 9, "reason": "a long stakeout"})
    assert "9.0 hour(s)" in result
    assert "day(s) advanced" in result

    reloaded = await store.load(campaign.id)
    # 9 hours = 3 steps of ~3h each; morning + 3 steps = dusk, no day rollover.
    assert reloaded.days_elapsed == 5
    assert reloaded.time_of_day == TimeOfDay.DUSK


@pytest.mark.asyncio
async def test_advance_time_rejects_non_positive_hours(store, campaign):
    tools = {t.name: t for t in make_travel_tools(campaign.id, store)}
    result = await tools["advance_time"].ainvoke({"hours": 0, "reason": "nothing"})
    assert "must be positive" in result


@pytest.mark.asyncio
async def test_take_rest_long_heals_and_restores(store, campaign):
    char = make_character("Tarvokk", current_hp=1, max_hp=12)
    campaign.party = [char]
    await store.save(campaign)

    tools = {t.name: t for t in make_travel_tools(campaign.id, store)}
    result = await tools["take_rest"].ainvoke({"kind": "long"})
    assert "long rest" in result.lower()

    reloaded = await store.load(campaign.id)
    assert reloaded.party[0].current_hp == 12


@pytest.mark.asyncio
async def test_take_rest_rejects_bad_kind(store, campaign):
    tools = {t.name: t for t in make_travel_tools(campaign.id, store)}
    result = await tools["take_rest"].ainvoke({"kind": "medium"})
    assert "must be 'short' or 'long'" in result


def test_time_guardrail_fires_on_narrated_skip_with_no_backing_tool_call():
    notes = "The party keeps watch overnight, and by dawn nothing has stirred."
    issue = _detect_missing_time_advance_followup(notes, called=set())
    assert issue is not None
    assert "advance_time" in issue


def test_time_guardrail_silent_when_advance_time_was_called():
    notes = "The party keeps watch overnight, and by dawn nothing has stirred."
    issue = _detect_missing_time_advance_followup(notes, called={"advance_time"})
    assert issue is None


def test_time_guardrail_silent_when_travel_to_was_called():
    notes = "By the time they arrive, dusk has fallen over the road."
    issue = _detect_missing_time_advance_followup(notes, called={"travel_to"})
    assert issue is None


def test_time_guardrail_silent_with_no_time_skip_language():
    notes = "Tarvokk swings his sword at the bandit and narrowly misses."
    issue = _detect_missing_time_advance_followup(notes, called=set())
    assert issue is None
