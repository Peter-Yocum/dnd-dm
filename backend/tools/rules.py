from langchain_core.tools import tool

from backend.stores.rules_store import RulesStore


def make_tools(rules_store: RulesStore, books_in_play: list[str]) -> list:
    @tool
    def search_rules(query: str) -> str:
        """Look up D&D 5e rules, spells, monsters, or items in the indexed rulebooks.
        Use for ANY rules question — how a spell works, a condition's effects, a
        monster's stat block, action economy, etc. Always cite the book, section,
        AND chunk_id from the results when relaying a fact. If the books don't
        cover it, say so and label any ruling as your own improvisation."""
        if not rules_store.is_ready():
            return (
                "Rulebook index is not ready. "
                "Run build_index.py first, then restart the app."
            )
        chunks = rules_store.search(query, books_in_play=books_in_play)
        if not chunks:
            return f"No relevant rules found for '{query}'."
        return "\n\n---\n\n".join(
            f"[{c.book} — {c.section} | chunk_id: {c.chunk_id}]\n{c.content}" for c in chunks
        )

    @tool
    def search_adventure_literal(query: str) -> str:
        """Literal, case-insensitive full-text search across every page of the
        adventure book(s) in play — NOT semantic/vector search like search_rules.
        Use this to find every mention of a specific named character, place, or
        item across the WHOLE book — e.g. checking whether an NPC introduced early
        on is referenced again in a later chapter (a reappearance, a hidden
        motivation only revealed later). search_rules's top-k similarity ranking
        can easily miss a single scattered forward-reference; this won't."""
        if not books_in_play:
            return "No adventure books in play to search (core rulebooks only)."
        chunks = rules_store.search_adventure_literal(query, books_in_play=books_in_play)
        if not chunks:
            return f"No mentions of '{query}' found in the adventure text."
        return "\n\n---\n\n".join(f"[{c.section}]\n{c.content}" for c in chunks)

    return [search_rules, search_adventure_literal]
