"""Renders a Location's grid into a template-friendly structure of colored
cells — the Maps browser's roguelike/MUD-style rendering (colored blocks
on a black background, not plain monospace text). Pure, no I/O, kept
separate from main.py so it's testable without a DB/HTTP round-trip.
"""

# Keyword -> CSS "kind" bucket. A small fixed palette (see static/style.css's
# .map-cell-* classes) rather than one color per arbitrary symbol — DM-authored
# legends use whatever vocabulary a scene calls for, so classification is by
# meaning (a legend's free-text description), not by the literal character.
_KIND_KEYWORDS: dict[str, list[str]] = {
    "wall": ["wall"],
    "door": ["door", "gate"],
    "water": ["water", "river", "lake", "swamp", "pool", "pond"],
    "vegetation": ["tree", "forest", "brush", "vegetation", "bush", "grass"],
    "rock": ["rock", "stone", "boulder", "cliff", "mountain"],
    "difficult": ["difficult", "rubble", "debris", "mud", "sand"],
    "furniture": ["crate", "barrel", "table", "chair", "furniture", "chest", "shelf"],
}


def classify_symbol(symbol: str, legend: dict[str, str]) -> str:
    """Maps a grid symbol to a small fixed 'kind' bucket used to pick a CSS
    color class. '.' is always 'floor'; a symbol whose legend text doesn't
    match any recognized keyword falls back to 'other' rather than guessing."""
    if symbol == ".":
        return "floor"
    meaning = legend.get(symbol, "").lower()
    for kind, keywords in _KIND_KEYWORDS.items():
        if any(kw in meaning for kw in keywords):
            return kind
    return "other"


def render_grid(grid: list[str], legend: dict[str, str]) -> list[list[dict[str, str]]]:
    """Returns rows of {"symbol": str, "kind": str} dicts, one per cell,
    ready for a template to render as colored `<span>`s. Empty input (no
    grid authored) returns an empty list — callers should check `grid`
    truthiness themselves to decide whether to render at all."""
    return [
        [{"symbol": ch, "kind": classify_symbol(ch, legend)} for ch in row]
        for row in grid
    ]


def render_grid_fogged(
    grid: list[str], legend: dict[str, str], revealed_positions: list[tuple[int, int]], radius: int = 2,
) -> list[list[dict[str, str]]]:
    """Player-facing fog-of-war version of render_grid — a cell farther than
    `radius` squares (Chebyshev distance, max(|dx|,|dy|) — same 5e
    diagonal-counts-as-5ft convention check_opportunity_attacks already
    uses, _helpers.py) from every entry in `revealed_positions` becomes a
    fog placeholder instead of its real symbol/kind. Same output shape as
    render_grid, so callers/templates need no changes beyond a
    '.map-cell-fog' CSS class. Never used for get_location_grid (the DM/
    mechanics model always gets the real grid) — this is strictly a Maps
    browser display concern.

    Caller's responsibility: only call this when revealed_positions is
    non-empty — an empty list here would fog the ENTIRE grid, which is the
    wrong default for a location nobody's fought in yet (see Location.
    revealed_positions' own docstring, models.py)."""
    revealed = list(revealed_positions)
    rows = render_grid(grid, legend)
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            in_view = any(max(abs(x - rx), abs(y - ry)) <= radius for rx, ry in revealed)
            if not in_view:
                cell["symbol"] = "?"
                cell["kind"] = "fog"
    return rows
