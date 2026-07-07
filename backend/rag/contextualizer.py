"""
ChunkContextualizer — Anthropic's "Contextual Retrieval" technique: prepend a
short LLM-generated situating blurb to a chunk before embedding/BM25 indexing,
so a fragment like "the innkeeper greeted them warmly" gets the name/place
context ("Toblen Stonehill of Phandalin's Stonehill Inn") folded into what
gets embedded, without changing the chunk's own citable text.

Ingest-time only (scripts/build_index.py) — never called on the query path.

Model choice: settings.mechanics_model (gemma4:26b-mlx), not a smaller/
different model, despite this being a high-volume bulk pass. This project
has a documented incident (see design.md) where a different, unvalidated
model (qwen2.5:14b) produced fake tool calls and garbled output under
sustained use; every other bulk LLM pass in this codebase (clean_source.py,
extract_entities.py) already defaults to gemma4:26b-mlx for the same
reliability reason. Override via --context-model if you want to experiment
with a faster model, but that's an explicit opt-in, not the default.
"""

from backend.config import settings


class ChunkContextualizer:
    def __init__(
        self,
        model: str = settings.mechanics_model,
        ollama_base_url: str = settings.ollama_base_url,
    ) -> None:
        self._model = model
        self._ollama_base_url = ollama_base_url

    def contextualize(self, chunk_text: str, parent_section_text: str, book: str) -> str:
        """Returns a 1-2 sentence situating blurb for chunk_text, given the
        larger parent_section_text it came from. Raises on a genuine LLM/
        connection failure — the caller (build_index.py) decides whether to
        skip-and-retry-later, matching extract_entities.py's per-item
        skip-on-doubt philosophy rather than aborting the whole run."""
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOllama(model=self._model, base_url=self._ollama_base_url, temperature=0, reasoning=False)
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
context — who/what/where it's about, using proper names from the section \
where relevant (e.g. "This describes the innkeeper Toblen Stonehill's \
Stonehill Inn in Phandalin"). Do not summarize the passage's content, only \
its context. Output ONLY that one sentence, nothing else."""),
        ])
        return response.content.strip()
