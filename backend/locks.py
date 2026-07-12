"""Per-campaign write locks, shared by EVERY layer that runs a
load→mutate→save cycle against the campaign store.

The store persists with delete-all/reinsert semantics, so the last writer
silently wins: two concurrent cycles — a LangGraph tool call, an HTTP
endpoint, a background task — can interleave load/save and drop one side's
mutation entirely. Confirmed live twice: parallel tool calls from a single
multi-tool AIMessage (ToolNode runs them via asyncio.gather), and
2026-07-10's vanishing loot, where session/end's slow summarize→save cycle
clobbered a tool-call save that landed mid-cycle.

Lived inside dm_agent.py (as a private _get_tool_lock) until 2026-07-11 —
but the HTTP layer had started importing the private symbol, and the
invariant this protects belongs to campaign persistence, not to the agent.
Any new writer must acquire this lock around its whole load→mutate→save
cycle. asyncio.Lock is NOT reentrant: never take it while already holding
it (helpers that save, e.g. main.py's _mint_active_thread, document
"caller owns the lock" instead of locking themselves).

Plain dict, no eviction: one Lock per campaign ever touched by this
process is a few hundred bytes each — negligible for a single-user app,
and correctness-critical entries must never disappear mid-hold.
"""

import asyncio

_locks: dict[str, asyncio.Lock] = {}


def campaign_write_lock(campaign_id: str) -> asyncio.Lock:
    return _locks.setdefault(campaign_id, asyncio.Lock())
