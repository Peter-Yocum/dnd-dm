"""
Deterministic fuzzy blocking for entity dedup — no LLM call. Used by
create_npc/create_location/add_item_to_character before inserting a new
entity, to catch a likely duplicate ("Toblen" vs "Toblen Stonehill") and
surface it as a warning rather than silently creating a duplicate or
silently merging. The actual dedup DECISION (is this really the same
entity?) stays LLM judgment — the calling model sees the warning and either
calls lookup_entity to confirm, or passes force=True to proceed anyway.
"""


def find_candidate_matches(name: str, candidates: list[str], threshold: float = 0.80) -> list[str]:
    """Returns the subset of `candidates` whose similarity to `name` meets
    `threshold` (0-1 scale). Pure Python/C-extension, no model download, no
    LLM call — deterministic blocking, not the dedup decision itself.

    Uses WRatio (rapidfuzz's weighted composite metric), not a plain edit-
    distance ratio: a plain token_sort_ratio scores "Toblen" vs "Toblen
    Stonehill" (a common real case — short form vs. full name) at only ~55,
    since it penalizes the length difference heavily, which would miss
    exactly the alias case this function exists to catch. WRatio scores
    that pair ~90 while still scoring unrelated names low (~25-35),
    verified directly against a handful of real name pairs before picking
    this metric/threshold combination."""
    if not name or not candidates:
        return []
    from rapidfuzz import fuzz  # lazy — keeps this module importable before rapidfuzz is installed

    name_lower = name.strip().lower()
    return [
        c for c in candidates
        if c.strip().lower() != name_lower
        and fuzz.WRatio(name_lower, c.strip().lower()) / 100.0 >= threshold
    ]
