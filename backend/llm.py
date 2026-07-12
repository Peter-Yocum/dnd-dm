"""Single construction point for every Ollama client in this codebase.

Every ChatOllama/OllamaEmbeddings in backend/ and scripts/ is built by one
of the two factories below — never construct one directly. Before this
module existed (2026-07-11) there were ~19 scattered construction sites,
and changing one client-level setting (a keep_alive experiment) meant
finding and touching all of them; the client_kwargs timeout fix of
2026-07-08 had already missed one site that way (rag/contextualizer.py,
which runs live in the request path via history_store.add_session).
Cross-cutting client policy — timeouts, keep_alive, reasoning — is decided
here once and documented here once.

Per-role choices stay at the call sites where they belong: temperature,
which model, and any constructor-injected base_url (the RAG/store classes
take base_url as a constructor param so scripts can point them at a
different Ollama; they pass it through here).
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings

from backend.config import settings

# Client-level httpx timeouts, in seconds. Confirmed live, 2026-07-08: with
# no timeout at all, an occasional hung Ollama request blocks its thread
# forever — a stuck socket read can't be interrupted by Python, even from
# asyncio.to_thread/wait_for (those only abandon the coroutine, not the
# thread) — permanently leaking one slot from the shared default thread-pool
# executor every time. Enough leaks (a handful of campaign-creation
# attempts) exhaust that pool and freeze the *whole app* for every user, not
# just the request that triggered it. This is also very likely the actual
# mechanism behind the "MLX runner stuck reporting 'Stopping...'
# indefinitely" hang chased for days before that.
#
# Chat gets 120s vs the embedders' 60s: a real multi-step tool-calling turn
# can legitimately take tens of seconds per call, while an embed normally
# completes in well under a second — 60s is already generous headroom there.
# Offline ingest scripts pass timeout=None instead (unbounded, their
# pre-factory behavior): an overnight batch on a busy machine can see
# legitimate multi-minute generations during model-swap thrash, and each
# script already bounds per-item damage with its own skip-on-doubt handling.
CHAT_TIMEOUT_S = 120.0
EMBED_TIMEOUT_S = 60.0


def ollama_chat(
    *,
    temperature: float = 0.0,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = CHAT_TIMEOUT_S,
) -> ChatOllama:
    """A ChatOllama with this app's cross-cutting client policy applied.

    reasoning=False on every instance (2026-07-04): gemma4:26b-mlx always
    wraps output in a <|channel>thought...<channel|> block, empty or not,
    whenever thinking isn't explicitly disabled, and langchain_ollama's
    `reasoning` default of None leaves any such tags embedded directly in
    `.content` instead of split into additional_kwargs — the source of a
    previously-investigated garbled-fragment leak into a Session 0 reply.
    No caller anywhere reads reasoning_content, so False (skip reasoning
    entirely) rather than True (perform it, capture it separately) — no
    product value in paying latency for reasoning nothing uses.

    keep_alive comes from settings.ollama_keep_alive (default None = the
    Ollama server's own idle-timeout eviction) — see that setting's comment
    in config.py for the 2026-07-10 keep_alive=-1 experiment and why forced
    residency was reverted.
    """
    return ChatOllama(
        model=model or settings.mechanics_model,
        base_url=base_url or settings.ollama_base_url,
        temperature=temperature,
        reasoning=False,
        keep_alive=settings.ollama_keep_alive,
        client_kwargs={"timeout": timeout} if timeout is not None else {},
    )


def ollama_embeddings(
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = EMBED_TIMEOUT_S,
) -> OllamaEmbeddings:
    """An OllamaEmbeddings with this app's cross-cutting client policy
    applied — same timeout/keep_alive reasoning as ollama_chat above, minus
    reasoning (chat-only param). Model defaults to settings.embed_model
    rather than being hardcoded per site (it was literal "nomic-embed-text"
    at five different call sites before this factory)."""
    return OllamaEmbeddings(
        model=model or settings.embed_model,
        base_url=base_url or settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive,
        client_kwargs={"timeout": timeout} if timeout is not None else {},
    )
