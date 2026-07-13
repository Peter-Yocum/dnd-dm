"""Single construction point for every chat/embedding client in this codebase.

Every ChatOllama/OllamaEmbeddings/ChatOpenAI in backend/ and scripts/ is
built by one of the factories below — never construct one directly. Before
this module existed (2026-07-11) there were ~19 scattered construction
sites, and changing one client-level setting (a keep_alive experiment)
meant finding and touching all of them; the client_kwargs timeout fix of
2026-07-08 had already missed one site that way (rag/contextualizer.py,
which runs live in the request path via history_store.add_session).
Cross-cutting client policy — timeouts, keep_alive, reasoning — is decided
here once and documented here once.

vllm_chat()/vllm_embeddings() (2026-07-13, vllm-migration-plan.md) are the
real construction points now — Ollama no longer serves anything in the
normal runtime path (chat AND embeddings both moved to vLLM-metal), which
is why this module's centralization already paid off twice: each swap was
one new factory function plus repointing a handful of call sites, not a
scattered hunt. ollama_chat() stays — offline scripts (extract_entities.py,
clean_source.py, add_headers.py) still use it, out of scope for both
migrations, and it's what the manual break-glass fallback would use.

Per-role choices stay at the call sites where they belong: temperature,
which model, and any constructor-injected base_url (the RAG/store classes
take base_url as a constructor param so scripts can point them at a
different Ollama; they pass it through here).
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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


def vllm_chat(
    *,
    temperature: float = 0.0,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = CHAT_TIMEOUT_S,
) -> ChatOpenAI:
    """A ChatOpenAI pointed at the vLLM-metal chat server (vllm-migration-plan.md).

    extra_body={"chat_template_kwargs": {"enable_thinking": False}} on every
    instance (2026-07-13, confirmed live during implementation, corrects an
    earlier wrong assumption in this docstring that Step 0's tool-forcing
    battery had already ruled this out — it hadn't tested a plain,
    no-tools-bound generation call): Qwen3-30B-A3B-4bit reasons by default,
    with the <think>...</think> block leaking straight into `.content`
    (confirmed: a plain "say OK" prompt came back as the full chain-of-
    thought plus "OK" appended) — the same class of problem `reasoning=False`
    solved for Ollama/Gemma4 (see ollama_chat's docstring and
    strip_reasoning_leakage in dm_agent.py), just a different tag format and
    a different disable mechanism. `enable_thinking=False` is Qwen3's own
    chat-template flag for skipping the reasoning pass entirely — confirmed
    live: with it set, `.content` came back clean (`"OK"`), `reasoning` was
    `null`, and completion_tokens dropped from 300 (truncated mid-reasoning
    at the max_tokens cap in one test) to 41. No caller anywhere reads
    reasoning content, so skip it entirely rather than route it to a
    separate field and pay the token/latency cost for nothing used — same
    philosophy as Ollama's reasoning=False, see that docstring's closing
    paragraph.

    No `keep_alive` kwarg — that was specifically Ollama's idle-eviction
    knob; there's no equivalent residency concept to configure for a
    directly-run vllm-metal process.

    api_key is a required field for ChatOpenAI/the OpenAI client library
    but meaningless here — vllm-metal doesn't check it, "unused" is a
    placeholder, not a real credential.
    """
    return ChatOpenAI(
        model=model or settings.mechanics_model,
        base_url=base_url or settings.vllm_base_url,
        api_key="unused",
        temperature=temperature,
        timeout=timeout,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def vllm_embeddings(
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = EMBED_TIMEOUT_S,
) -> OpenAIEmbeddings:
    """An OpenAIEmbeddings pointed at the vLLM-metal embed server
    (vllm-migration-plan.md §7.7) — was OllamaEmbeddings/nomic-embed-text
    until this migration. Model defaults to settings.embed_model rather
    than being hardcoded per site (it was literal "nomic-embed-text" at
    five different call sites before the original ollama_embeddings()
    factory this replaces).

    Real, live-verified reason for the swap, not just consolidation:
    nomic-embed-text's architecture (NomicBertModel, a BERT-family encoder)
    cannot be served via vllm-metal at all — vllm-metal delegates model
    loading entirely to mlx_lm, which is a causal-LM-only library (confirmed
    by listing every architecture file in mlx_lm/models — zero encoder
    models exist there). mlx-community/Qwen3-Embedding-0.6B-8bit, served via
    `vllm serve ... --convert embed` (vLLM's adapter for repurposing a
    causal generation model as a pooling/embedding model), is the verified
    alternative: a real /v1/embeddings request returned correct 1024-dim
    vectors in testing.

    api_key is a required field for OpenAIEmbeddings/the OpenAI client
    library but meaningless here — vllm-metal doesn't check it, "unused" is
    a placeholder, not a real credential.
    """
    return OpenAIEmbeddings(
        model=model or settings.embed_model,
        base_url=base_url or settings.vllm_embed_base_url,
        api_key="unused",
        timeout=timeout,
    )
