"""BDD-style coverage for the mechanics/narrator reliability fixes in
backend/agent/dm_agent.py — the generalized "VERIFIED STATE CHANGES" relay,
and the guardrails that catch a narrated outcome (a kill, combat ending)
with no real tool call behind it. Built directly from a live incident: three
captured goblins narrated as executed while remaining alive in the database,
with the stale active_encounter silently mis-gating every turn afterward.

These test the pure detector functions/regexes, not the LLM itself (there's
no way to unit-test what a model will actually write) — they guard against
regressing the specific patterns that slipped through before.
"""
import pytest
from langchain_core.messages import ToolMessage

from backend.agent.dm_agent import (
    _COMBAT_ENDED_MENTION_RE, _LOOKUP_ONLY_TOOLS, _detect_missing_monster_death_followup,
    _verified_state_changes_note,
)


def test_combat_end_detection_catches_capture_not_just_a_killing_blow():
    # Given narration describing prisoners taken, not enemies slain
    captured = "The three goblins are bound and captured, no longer a threat."
    surrendered = "the goblins surrendered and are now restrained"
    explicit = "the fight is over"
    unrelated = "the party continues down the corridor"

    # Then the capture/surrender phrasing is recognized as combat ending too
    assert _COMBAT_ENDED_MENTION_RE.search(captured)
    assert _COMBAT_ENDED_MENTION_RE.search(surrendered)
    # And the original explicit phrasing still works
    assert _COMBAT_ENDED_MENTION_RE.search(explicit)
    # And ordinary narration doesn't false-positive
    assert not _COMBAT_ENDED_MENTION_RE.search(unrelated)


def test_narrated_kill_without_hp_tool_call_is_flagged():
    # Given a resolution report narrating an execution
    notes = "Tarvokk slits their throat, ending the goblin for good."

    # When no update_monster_hp/resolve_attack/end_encounter backs it up
    issue = _detect_missing_monster_death_followup(notes, called=set())
    # Then the guardrail fires
    assert issue is not None
    assert "update_monster_hp" in issue

    # But once the real tool call happens, the same notes are cleared
    assert _detect_missing_monster_death_followup(notes, called={"update_monster_hp"}) is None


def test_narration_with_no_kill_language_never_trips_the_guardrail():
    notes = "The party continues down the corridor, weapons still sheathed."
    assert _detect_missing_monster_death_followup(notes, called=set()) is None


def test_verified_state_changes_captures_mutations_and_skips_lookups():
    # Given a turn where a real HP change happened alongside a pure lookup call
    scratch = [
        ToolMessage(content="Goblin 1: 1 -> 0 HP -- DEFEATED", name="update_monster_hp", tool_call_id="a"),
        ToolMessage(content="Party status: Tarvokk 12/12 HP", name="get_party_status", tool_call_id="b"),
    ]

    note = _verified_state_changes_note(scratch)

    # Then the real mutation is relayed verbatim to the narrator...
    assert "Goblin 1: 1 -> 0 HP -- DEFEATED" in note
    # ...but the read-only lookup is not, since it's not a state change
    assert "Party status" not in note


def test_verified_state_changes_note_is_empty_when_nothing_mutated():
    scratch = [
        ToolMessage(content="Party status: all fine", name="get_party_status", tool_call_id="a"),
    ]
    assert _verified_state_changes_note(scratch) == ""


def test_lookup_only_tools_is_the_small_exclusion_list_not_an_allowlist():
    # The whole point of this list is that it's short and exclusion-based —
    # a new mutating tool added later is captured by default rather than
    # silently falling through the same hole this mechanism exists to close.
    assert "get_party_status" in _LOOKUP_ONLY_TOOLS
    assert "update_monster_hp" not in _LOOKUP_ONLY_TOOLS
    assert "resolve_attack" not in _LOOKUP_ONLY_TOOLS
