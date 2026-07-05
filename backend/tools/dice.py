from langchain_core.tools import tool

from backend.tools._helpers import roll_notation


def make_tools() -> list:
    @tool
    def roll_dice(notation: str) -> str:
        """Roll dice in standard notation: 'd20', '1d20+5', '2d6-1', '4d6kh3' (keep
        highest 3). Fallback for any roll with no dedicated resolution tool — prefer
        resolve_attack/resolve_saving_throw/resolve_check/cast_spell whenever the
        roll is tied to a specific character's or monster's stat block, since those
        also apply the result automatically. Use this directly for flavor rolls,
        tables, and environmental checks not tied to anyone's sheet. Never invent a
        number; always call this tool."""
        try:
            total, breakdown = roll_notation(notation)
            return f"{notation}: {breakdown}"
        except ValueError as e:
            return f"Error: {e}"

    return [roll_dice]
