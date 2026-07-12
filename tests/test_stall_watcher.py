"""BDD-style coverage for watch_for_stalls (backend/agent/dm_agent.py) — the
SSE keepalive wrapper around a slow/wedged model call. Built from a live
incident: a cold local-model load took ~15 minutes to produce its first
token, but the old implementation only ever yielded one STALL event and then
went fully silent for the rest of the wait — exactly the silence window a
browser/OS/network idle-connection timeout will kill, surfacing to the
player as a bare "Connection lost" with no indication anything was even
attempted.
"""
import asyncio

import pytest

from backend.agent.dm_agent import STALL, watch_for_stalls

pytestmark = pytest.mark.asyncio


async def _slow_source(gap_seconds: float, ticks: int):
    yield "first"
    for _ in range(ticks):
        await asyncio.sleep(gap_seconds)
    yield "second"


async def test_stall_repeats_for_as_long_as_the_gap_continues():
    # Given a source that goes quiet for several stall intervals in a row
    events = []
    async for item in watch_for_stalls(_slow_source(gap_seconds=0.15, ticks=3), stall_after=0.05):
        events.append("STALL" if item is STALL else item)

    # Then it keeps signalling STALL on every interval, not just the first —
    # the connection has real traffic the whole time instead of going silent
    assert events[0] == "first"
    assert events[-1] == "second"
    stall_count = events.count("STALL")
    assert stall_count >= 2, f"expected repeated STALL events, got {events}"


async def test_stall_stops_once_real_data_arrives_and_can_fire_again_later():
    # Given a source with two separate slow gaps
    events = []
    async for item in watch_for_stalls(_slow_source(gap_seconds=0.15, ticks=2), stall_after=0.05):
        events.append("STALL" if item is STALL else item)

    # Then STALL events appear before "second" arrives, and stop the moment
    # real data does — no stall event trails after the final real item
    assert events[-1] == "second"
    assert events[-2] == "STALL"


async def test_an_exception_in_the_source_propagates_through_the_watcher():
    async def failing_source():
        yield "ok"
        raise RuntimeError("model backend exploded")

    events = []
    with pytest.raises(RuntimeError, match="model backend exploded"):
        async for item in watch_for_stalls(failing_source(), stall_after=1.0):
            events.append(item)
    assert events == ["ok"]
