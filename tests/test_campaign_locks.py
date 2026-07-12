"""BDD-style coverage for the per-campaign reader/writer lock (backend/locks.py)
and its use in _make_tool_node (backend/agent/dm_agent.py) — built from a live
incident: a single shared mutex meant one hung read-only tool call
(search_rules) made two other, genuinely-instant read-only calls in the same
batch (get_campaign_summary, get_current_location) queue up behind it and
look equally frozen, even though only one tool call actually hung.
"""
import asyncio
import uuid

import pytest

import backend.agent.dm_agent as dm_agent
from backend.locks import campaign_read_lock, campaign_write_lock

pytestmark = pytest.mark.asyncio


# ── _CampaignRWLock direct coverage ─────────────────────────────────────────

async def test_two_readers_proceed_concurrently_without_waiting_on_each_other():
    cid = str(uuid.uuid4())
    order: list[str] = []

    async def reader(name: str, hold_seconds: float):
        async with campaign_read_lock(cid):
            order.append(f"{name}-start")
            await asyncio.sleep(hold_seconds)
            order.append(f"{name}-end")

    # Given two readers, one much slower than the other
    await asyncio.gather(reader("slow", 0.15), reader("fast", 0.01))

    # Then the fast one starts AND finishes before the slow one finishes —
    # it never waited behind the slow reader holding the same lock
    assert order.index("fast-end") < order.index("slow-end")
    assert order.index("slow-start") < order.index("fast-end")


async def test_a_writer_waits_for_an_active_reader_to_fully_exit():
    cid = str(uuid.uuid4())
    events: list[str] = []

    async def reader():
        async with campaign_read_lock(cid):
            events.append("read-start")
            await asyncio.sleep(0.1)
            events.append("read-end")

    async def writer():
        await asyncio.sleep(0.02)  # let the reader get in first
        async with campaign_write_lock(cid):
            events.append("write-start")

    await asyncio.gather(reader(), writer())

    # The writer never starts until the reader has fully released
    assert events.index("write-start") > events.index("read-end")


async def test_a_new_reader_waits_while_a_writer_is_active_then_resumes():
    cid = str(uuid.uuid4())
    events: list[str] = []

    async def writer():
        async with campaign_write_lock(cid):
            events.append("write-start")
            await asyncio.sleep(0.1)
            events.append("write-end")

    async def reader():
        await asyncio.sleep(0.02)  # let the writer acquire first
        async with campaign_read_lock(cid):
            events.append("read-start")

    await asyncio.gather(writer(), reader())

    assert events.index("read-start") > events.index("write-end")


# ── _make_tool_node integration ─────────────────────────────────────────────

async def _call_wrapped(node, tool_name: str, execute):
    """Invoke the private awrap_tool_call wrapper directly with a minimal
    fake request — exercises the exact lock-selection/timeout logic without
    needing a full LangGraph invocation."""
    class _FakeRequest:
        tool_call = {"name": tool_name}

    return await node._awrap_tool_call(_FakeRequest(), execute)


async def test_a_slow_lookup_tool_does_not_block_a_fast_lookup_tool():
    cid = str(uuid.uuid4())
    node = dm_agent._make_tool_node([], cid)
    order: list[str] = []

    async def slow_read(req):
        order.append("slow-start")
        await asyncio.sleep(0.15)
        order.append("slow-end")
        return "slow"

    async def fast_read(req):
        order.append("fast-start")
        order.append("fast-end")
        return "fast"

    # "search_rules" and "get_campaign_summary" are both real _LOOKUP_ONLY_TOOLS
    await asyncio.gather(
        _call_wrapped(node, "search_rules", slow_read),
        _call_wrapped(node, "get_campaign_summary", fast_read),
    )

    # The fast lookup finished without waiting behind the slow one — this is
    # the exact incident: a hung search_rules must not stall an unrelated,
    # otherwise-instant read in the same batch
    assert order.index("fast-end") < order.index("slow-end")


async def test_a_mutating_tool_still_gets_exclusive_access(monkeypatch):
    cid = str(uuid.uuid4())
    node = dm_agent._make_tool_node([], cid)
    events: list[str] = []

    async def slow_write(req):
        events.append("write-start")
        await asyncio.sleep(0.1)
        events.append("write-end")
        return "written"

    async def read_after(req):
        await asyncio.sleep(0.02)
        events.append("read-start")
        return "read"

    # "update_monster_hp" is a real mutating tool name, "get_current_location"
    # a real lookup one
    await asyncio.gather(
        _call_wrapped(node, "update_monster_hp", slow_write),
        _call_wrapped(node, "get_current_location", read_after),
    )

    assert events.index("read-start") > events.index("write-end")


async def test_a_tool_call_that_hangs_past_the_timeout_raises_instead_of_hanging_forever(monkeypatch):
    # This exercises _serialize (the awrap_tool_call wrapper) directly,
    # bypassing LangGraph's own ToolNode._arun_one — real end-to-end
    # confirmation that ToolNode's `try/except Exception` around
    # `await self._awrap_tool_call(...)` (verified by reading the installed
    # langgraph.prebuilt.tool_node source) converts whatever this wrapper
    # raises into a normal error ToolMessage lives in langgraph itself, not
    # in this codebase, so it isn't re-tested here — what matters at this
    # layer is that a wedged tool body produces a bounded, catchable
    # TimeoutError instead of hanging indefinitely.
    cid = str(uuid.uuid4())
    node = dm_agent._make_tool_node([], cid)
    monkeypatch.setattr(dm_agent, "_TOOL_EXECUTION_TIMEOUT_S", 0.05)

    async def wedged(req):
        await asyncio.sleep(10)
        return "never gets here"

    with pytest.raises(TimeoutError):
        await _call_wrapped(node, "search_rules", wedged)
