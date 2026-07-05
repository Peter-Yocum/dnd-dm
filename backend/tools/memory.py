from langchain_core.tools import tool

from backend.stores.history_store import HistoryStore


def make_tools(history_store: HistoryStore, campaign_id: str) -> list:
    @tool
    def search_campaign_history(query: str) -> str:
        """Search the chronicles of past sessions for relevant events, NPC
        encounters, revelations, or decisions made by the party.

        Use this when:
        - A player references something from an earlier session
        - A previously-met NPC reappears
        - The party is following up on a past plot hook
        - You need context about a location the party has visited before

        Do NOT use this for rules questions — use search_rules instead."""
        results = history_store.search(query, campaign_id=campaign_id)
        if not results:
            return "No relevant past session records found for this campaign."
        return "\n\n---\n\n".join(
            f"[Session {doc.metadata.get('session_number', '?')}]\n{doc.page_content}"
            for doc in results
        )

    return [search_campaign_history]
