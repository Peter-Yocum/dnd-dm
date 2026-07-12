"""
Sufficiency grading + query reformulation for search_lore's bounded internal
retry (Stage 2's CRAG/Self-RAG pattern): retrieve -> grade -> (if
insufficient) reformulate & re-retrieve once -> return. Cheap local
LLM-as-judge calls — this runs on the query path (unlike Stage 0's ingest-
time contextualization), so it must stay fast, hence settings.mechanics_model
rather than anything heavier, and a single bounded retry rather than a loop.
"""

from backend.config import settings
from backend.stores.rules_store import RuleChunk


def grade_sufficiency(query: str, chunks: list[RuleChunk], model: str = settings.mechanics_model) -> bool:
    """Structured yes/no: do these chunks plausibly answer the query? Cheap,
    temp=0, single short call. Skip-on-doubt: a malformed/unparseable
    response is treated as sufficient (don't force a retry we can't judge
    the need for) rather than looping forever."""
    if not chunks:
        return False

    from langchain_core.messages import HumanMessage, SystemMessage

    from backend.llm import ollama_chat

    llm = ollama_chat(model=model)
    excerpt = "\n\n---\n\n".join(c.content[:500] for c in chunks[:5])
    response = llm.invoke([
        SystemMessage(content=(
            "You judge whether retrieved passages are sufficient to answer a "
            "query. Answer with exactly one word: YES or NO."
        )),
        HumanMessage(content=f"""Query: {query}

Retrieved passages:
{excerpt}

Do these passages plausibly contain enough information to answer the query? \
Answer YES or NO, nothing else."""),
    ])
    answer = response.content.strip().upper()
    return not answer.startswith("NO")


def reformulate_query(query: str, model: str = settings.mechanics_model) -> str:
    """One alternate phrasing of the query — widens the net for a re-retrieve
    when the first pass was judged insufficient. Falls back to the original
    query unchanged on any parse/generation failure."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from backend.llm import ollama_chat

    llm = ollama_chat(model=model)
    response = llm.invoke([
        SystemMessage(content=(
            "You rewrite a search query as one alternate phrasing that might "
            "surface different relevant results — synonyms, a more specific "
            "or more general framing, or a different angle on the same "
            "question. Output ONLY the rewritten query, nothing else."
        )),
        HumanMessage(content=f"Original query: {query}\n\nRewritten query:"),
    ])
    rewritten = response.content.strip().strip('"')
    return rewritten if rewritten else query
