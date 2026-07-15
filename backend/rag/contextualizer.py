"""
ChunkContextualizer — Anthropic's "Contextual Retrieval" technique: prepend a
short LLM-generated situating blurb to a chunk before embedding/BM25 indexing,
so a fragment like "the innkeeper greeted them warmly" gets its actual
name/place context (whatever NPC/location the surrounding section actually
names) folded into what gets embedded, without changing the chunk's own
citable text.

2026-07-13: the prompt used to give a concrete worked example using real
Lost Mine of Phandelver proper nouns ("Toblen Stonehill's Stonehill Inn in
Phandalin") — confirmed live that the model would sometimes echo that exact
example verbatim into unrelated core-rulebook passages (a generic PHB
combat-rules chunk got contextualized as being about "the Stonehill Inn"),
i.e. pattern-matching onto the prompt's own illustration instead of
grounding in the actual content. The prompt below no longer gives a
memorable concrete example — it explicitly forbids inventing/borrowing a
name not present in the actual passage, with a neutral fallback for
generic passages.

Ingest-time only (scripts/build_index.py) — never called on the query path.

Model choice: settings.mechanics_model, not a smaller/different model,
despite this being a high-volume bulk pass. This project has a documented
incident (see design.md) where a different, unvalidated model
(qwen2.5:14b) produced fake tool calls and garbled output under sustained
use; every other bulk LLM pass in this codebase (clean_source.py,
extract_entities.py) already defaults to the same validated model for the
same reliability reason. Override via --context-model if you want to
experiment with a faster model, but that's an explicit opt-in, not the
default.

Client: vllm_chat() (2026-07-13, vllm-migration-plan.md) — was
ollama_chat() until the vLLM-metal migration.
"""

from backend.config import settings


class ChunkContextualizer:
    def __init__(
        self,
        model: str = settings.mechanics_model,
        vllm_base_url: str = settings.vllm_base_url,
    ) -> None:
        self._model = model
        self._vllm_base_url = vllm_base_url
        # Built ONCE, reused across every contextualize() call (2026-07-13) —
        # this used to construct a fresh vllm_chat()/ChatOpenAI (and thus a
        # fresh underlying httpx client + TCP connection through Docker's
        # host-networking layer) on every single call. Confirmed live: once
        # build_index.py started firing several contextualize() calls
        # concurrently per batch (a thread pool, one per child chunk), most
        # of each batch's wall time was neither the model generating
        # (confirmed via the server's own throughput log) nor the DB
        # (confirmed idle via pg_stat_activity) — it was unaccounted-for gap
        # consistent with re-establishing a fresh connection per call instead
        # of reusing a warm pool. httpx/the OpenAI SDK's client are
        # documented thread-safe for exactly this kind of concurrent reuse.
        from backend.llm import vllm_chat

        self._llm = vllm_chat(model=self._model, base_url=self._vllm_base_url)

    def contextualize(self, chunk_text: str, parent_section_text: str, book: str) -> str:
        """Returns a 1-2 sentence situating blurb for chunk_text, given the
        larger parent_section_text it came from. Raises on a genuine LLM/
        connection failure — the caller (build_index.py) decides whether to
        skip-and-retry-later, matching extract_entities.py's per-item
        skip-on-doubt philosophy rather than aborting the whole run."""
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = self._llm
        # Cap the parent section handed to the model — it's context for the
        # blurb, not something that needs to be reproduced or fully quoted.
        parent_excerpt = parent_section_text[:4000]
        response = llm.invoke([
            SystemMessage(content=(
                "You write a short situating blurb for a passage from a D&D "
                f"rulebook ({book}), so the passage can be understood on its "
                "own once separated from its surrounding text."
            )),
            HumanMessage(content=f"""Full section this passage comes from:
{parent_excerpt}

Passage to situate:
{chunk_text}

Write ONE short sentence (max ~25 words) that gives this passage's key \
context — who/what/where it's about. Use ONLY names/places that actually \
appear in the section above — never invent or borrow a name from anywhere \
else, and if the passage is generic rules text with no specific named \
character/place, say so generically (e.g. "This describes the rules for \
{{topic}}") rather than inventing one. Do not summarize the passage's \
content, only its context. Output ONLY that one sentence, nothing else."""),
        ])
        return response.content.strip()
