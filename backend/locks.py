"""Per-campaign read/write locks, shared by EVERY layer that runs a
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

A REAL reader/writer lock, not a single asyncio.Lock — confirmed live,
2026-07-12: a single mutex forces even read-only tool calls (search_rules,
get_campaign_summary, get_current_location) to serialize behind each other
AND behind writes, so a single hung read-only call queued up two other,
otherwise-instant read-only calls behind it (an incident that looked like
"everything froze" but was really one stuck call plus two innocent
bystanders waiting on the same lock). Any number of readers now run
concurrently; a writer gets exclusive access (waits for active readers to
drain, blocks new readers/writers until done); readers resume freely once
the writer finishes. Classic "first reader takes the write-lock, last
reader releases it" construction — no reader/writer primitive exists in
the asyncio stdlib.

Any new writer must acquire `campaign_write_lock` around its whole
load→mutate→save cycle; a read-only operation that just needs a consistent
snapshot (not a mutation) should use `campaign_read_lock` instead. Neither
is reentrant: never take either while already holding one (helpers that
save, e.g. main.py's _mint_active_thread, document "caller owns the lock"
instead of locking themselves).

Plain dict, no eviction: one lock-pair per campaign ever touched by this
process is a few hundred bytes each — negligible for a single-user app, and
correctness-critical entries must never disappear mid-hold.
"""

import asyncio

_locks: dict[str, "_CampaignRWLock"] = {}


class _CampaignRWLock:
    def __init__(self) -> None:
        self._readers = 0
        self._readers_count_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    def writer(self) -> asyncio.Lock:
        """Exclusive access — `async with` directly."""
        return self._write_lock

    def reader(self) -> "_ReadLockCtx":
        """Shared access — any number of readers may hold this
        concurrently; only blocks while a writer is active."""
        return _ReadLockCtx(self)


class _ReadLockCtx:
    def __init__(self, rw: _CampaignRWLock) -> None:
        self._rw = rw

    async def __aenter__(self) -> None:
        async with self._rw._readers_count_lock:
            self._rw._readers += 1
            if self._rw._readers == 1:
                await self._rw._write_lock.acquire()

    async def __aexit__(self, *exc_info: object) -> None:
        async with self._rw._readers_count_lock:
            self._rw._readers -= 1
            if self._rw._readers == 0:
                self._rw._write_lock.release()


def _get_rwlock(campaign_id: str) -> _CampaignRWLock:
    return _locks.setdefault(campaign_id, _CampaignRWLock())


def campaign_write_lock(campaign_id: str) -> asyncio.Lock:
    """Exclusive lock for a load→mutate→save cycle. `async with
    campaign_write_lock(campaign_id):` — same call signature as before this
    module gained real read/write distinction, so existing callers are
    unaffected."""
    return _get_rwlock(campaign_id).writer()


def campaign_read_lock(campaign_id: str) -> _ReadLockCtx:
    """Shared lock for a read-only operation against a campaign — any
    number of readers proceed concurrently; only waits while a writer
    currently holds `campaign_write_lock` for the same campaign_id.
    `async with campaign_read_lock(campaign_id):`"""
    return _get_rwlock(campaign_id).reader()
