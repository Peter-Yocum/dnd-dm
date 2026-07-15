"""
DraftStore — in-memory character drafts during Session 0 creation.

One draft per (campaign_id, player_slug). Lost on server restart, which is
fine: if the browser session is gone the draft is meaningless. Finalized
characters are persisted to Postgres via the campaign store.
"""


def _empty() -> dict:
    return {
        "name": "",
        "race": "",
        "subrace": "",
        "char_class": "",
        "subclass": "",
        "background": "",
        "alignment": "",
        "appearance": "",
        "pronouns": "",
        "strength": 0,
        "dexterity": 0,
        "constitution": 0,
        "intelligence": 0,
        "wisdom": 0,
        "charisma": 0,
        "skill_proficiencies": [],
        "spells_known": [],
        "personality_traits": [],
        "ideals": [],
        "bonds": [],
        "flaws": [],
        "backstory": "",
        "notes": "",
    }


class DraftStore:
    """Module-level singleton. Import and use `draft_store` directly."""

    def __init__(self) -> None:
        self._drafts: dict[str, dict] = {}

    def _key(self, campaign_id: str, player_slug: str) -> str:
        return f"{campaign_id}:{player_slug}"

    def get(self, campaign_id: str, player_slug: str) -> dict:
        key = self._key(campaign_id, player_slug)
        if key not in self._drafts:
            self._drafts[key] = _empty()
        return self._drafts[key]

    def update(self, campaign_id: str, player_slug: str, field: str, value: str) -> str:
        """Set a field on the draft. Returns a human-readable confirmation."""
        draft = self.get(campaign_id, player_slug)

        int_fields = {"strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"}
        list_fields = {"skill_proficiencies", "spells_known", "personality_traits", "ideals", "bonds", "flaws"}

        if field in int_fields:
            try:
                draft[field] = int(value)
                return f"Set {field} to {value}."
            except ValueError:
                return f"Error: '{value}' is not a valid integer for {field}."
        elif field in list_fields:
            # Allow comma-separated or single value; prefix CLEAR to reset
            if value.upper().startswith("CLEAR"):
                draft[field] = []
                return f"Cleared {field}."
            items = [v.strip() for v in value.split(",") if v.strip()]
            draft[field] = items
            return f"Set {field} to: {', '.join(items)}."
        elif field in draft:
            draft[field] = value
            return f"Set {field} to '{value}'."
        else:
            return f"Unknown field '{field}'. Valid fields: {', '.join(_empty().keys())}."

    def clear(self, campaign_id: str, player_slug: str) -> None:
        self._drafts.pop(self._key(campaign_id, player_slug), None)


# Singleton used by tools and routes
draft_store = DraftStore()
