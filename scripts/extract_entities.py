#!/usr/bin/env python3
"""
extract_entities.py — Precompute a structured NPC/Location/Item entity index
for an adventure, once per book, reused by every campaign's world-prep and by
the live DM agent's create_npc/create_location/add_item_to_character tools.

Root problem this exists to fix: a single live agent call over a whole
chapter's text has to both DISCOVER every named entity and SYNTHESIZE a full
profile for each in one pass. Tested live against Curse of Strahd: this
missed 4 of 5 expected NPCs — everyone whose only evidence was mentions
scattered across the whole document, as opposed to one concentrated scene.
None of this is campaign-specific (the same book produces the same
roster/gazetteer for every campaign that uses it), so it's computed once
here rather than re-derived live per campaign.

Pipeline:
  1. Windowed discovery  — one combined pass per ~20k-char window asking for
     every named person, place, AND item mentioned, tagged by type. Small
     windows make exhaustive enumeration reliable; a single pass over the
     whole book is exactly the failure mode this replaces.
  2. Canonicalize         — merge aliases/variant forms across all windows
     into one registry per adventure (e.g. "Rose" / "Rosavalda" / full
     name -> one canonical entity).
  3. Reference collation  — for each canonical name, gather every literal
     mention anywhere in the book. Standalone regex scan over the raw
     markdown — no RulesStore/Chroma/DB dependency, matching every other
     ingestion-time script's lack of a live-app dependency.
  4. Profile generation   — type-specific (see NPCExtractor/LocationExtractor
     below); Locations also get a separate connections-generation pass,
     since a relationship graph is a structurally different extraction
     than a flat per-entity profile.
  5. Write docs/source/adventures/{slug}/_entities.json, checkpointing after
     each entity so an interrupted run doesn't lose completed work (same
     reasoning as clean_source.py/add_headers.py's per-item checkpointing).

Extending to a new entity type (Faction, Quest, ...) means writing a new
EntityExtractor subclass and registering it in EXTRACTORS below — the
orchestration in extract_book() never branches on entity type itself. Item
is one such extension (see ItemExtractor); Monster is another, added for
core-rulebook extraction (see MonsterExtractor and --source-type below).

Usage:
    python extract_entities.py --book "Curse of Strahd"
    python extract_entities.py --book "Curse of Strahd" --dry-run
    python extract_entities.py --book "Curse of Strahd" --force
    python extract_entities.py --model qwen2.5:14b --ollama-url http://gaming-rig:11434
    python extract_entities.py --book "D&D 5E - Monster Manual" --source-type core
"""

import argparse
import asyncio
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path

# Windows' default console codec (cp1252/"charmap") can't encode the em-dashes/
# box-drawing/arrow characters this script prints for readability — confirmed
# live as a real UnicodeEncodeError crash on a fresh Windows venv (Mac/Linux
# default to UTF-8 stdout so this never surfaced there). errors="replace"
# rather than "strict" so a genuinely unencodable character degrades to "?"
# instead of crashing an otherwise-successful entity mid-run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_INPUT  = "docs/source/adventures"
DEFAULT_MODEL  = "gemma4:26b-mlx"
DEFAULT_OLLAMA = "http://localhost:11434"

# Real 5e stat blocks always print "Armor Class" immediately followed (within
# one short type/alignment line) by "Hit Points" — verified directly against
# a real monster entry (Bugbear: "Armor Class 16 ..." then "Hit Points 27
# ..." six lines later, ~120 chars apart). Requiring BOTH markers, WITHIN a
# small character gap, is what actually distinguishes a genuine stat block
# from a false positive — checked live against Lost Mine of Phandelver:
# "+1 Armor" mentions "Armor Class" (describing its bonus) but never "Hit
# Points" at all (caught by requiring both); "RULES INDEX" mentions both
# terms as separate alphabetized index entries thousands of characters apart
# (caught by the gap check, not by requiring both alone). A plain "look for
# 'Armor Class' anywhere in the section" first cut caught both of these.
_STAT_BLOCK_AC_RE = re.compile(r"\bArmor Class\b", re.IGNORECASE)
_STAT_BLOCK_HP_RE = re.compile(r"\bHit Points\b", re.IGNORECASE)
_STAT_BLOCK_MAX_AC_HP_GAP = 300  # chars

# Numbered site/area entries ("11. Old Garrison", "5. Overpass") are
# locations, not monsters, even on the rare chance their room description
# happens to mention a defender's AC/HP close together — exclude by header
# shape as an extra safety net alongside the AC/HP proximity check.
_NUMBERED_HEADER_RE = re.compile(r"^\d+[.)]\s")

# Tuned for reliable exhaustive name enumeration, not RAG chunk size — much
# bigger than build_index.py's ~1500-char chunks, since discovery needs
# whole scenes in view, not search-sized fragments.
DISCOVERY_WINDOW_CHARS   = 20_000
REFERENCE_CONTEXT_CHARS  = 300   # chars of context kept on each side of a literal match
MAX_REFERENCES_PER_NAME  = 20    # cap per-entity reference collation, same spirit as rules_store.py's search_adventure_literal limit

# Locations get a wider net than NPCs: a site's sub-areas ("A2f. The
# Basement") don't reliably repeat the site's own name the way a person's
# name recurs at every mention of them, so a plain literal-match collation
# undershoots for locations specifically unless given more room per hit.
# Items start with NPC's defaults — a named item is usually introduced once
# and then referred to by pronoun/generic term, so a modest window is enough.
REFERENCE_PARAMS = {
    "npc":      {"context_chars": REFERENCE_CONTEXT_CHARS, "limit": MAX_REFERENCES_PER_NAME},
    "location": {"context_chars": 800,                     "limit": 30},
    "item":     {"context_chars": REFERENCE_CONTEXT_CHARS, "limit": MAX_REFERENCES_PER_NAME},
}

_HEADER_RE = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)   # matches build_index.py's chunker boundary


# ── windowing ────────────────────────────────────────────────────────────────

def _read_book_text(book: str, input_dir: str, source_type: str = "adventure") -> str:
    """Concatenate this book's source text into one blob. Adventures are a
    subfolder of one-or-more .md files (docs/source/adventures/{slug}/*.md);
    core rulebooks are a single flat .md file directly under docs/source/core/
    (no subfolder per book) — read it directly instead of globbing."""
    if source_type == "core":
        path = Path(f"{input_dir}/{book}.md")
        return path.read_text(encoding="utf-8") if path.exists() else ""
    parts = []
    for path in sorted(Path(f"{input_dir}/{book}").glob("*.md")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split on #/##/### headers into (header_title, section_text) pairs —
    deterministic, no LLM call. Used for monster stat-block candidates,
    where each header-bounded section is already an unambiguous unit (unlike
    NPC/Location/Item names scattered through prose)."""
    matches = list(_HEADER_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(1).strip(), text[m.start():end]))
    return sections


def discover_monster_candidates(book_text: str) -> dict[str, str]:
    """Returns {monster_name: section_text} — deterministic, no LLM call and
    no canonicalization: a Monster Manual entry's name is already canonical
    (no "Rose"/"Rosavalda"-style aliasing ambiguity the way NPC prose has),
    so monster extraction skips discover_candidates/canonicalize entirely
    and goes straight from header-bounded section to profile generation.
    See _STAT_BLOCK_AC_RE's comment for why both AC+HP, close together, is
    the real signal — not just "Armor Class" appearing anywhere."""
    candidates: dict[str, str] = {}
    for title, section_text in _split_into_sections(book_text):
        if len(title) < 3 or _NUMBERED_HEADER_RE.match(title):
            continue
        ac_match = _STAT_BLOCK_AC_RE.search(section_text)
        hp_match = _STAT_BLOCK_HP_RE.search(section_text)
        if not ac_match or not hp_match:
            continue
        if abs(hp_match.start() - ac_match.start()) > _STAT_BLOCK_MAX_AC_HP_GAP:
            continue
        candidates[title] = section_text
    return candidates


def _split_into_windows(text: str, max_chars: int = DISCOVERY_WINDOW_CHARS) -> list[str]:
    """Split on #/##/### headers (same boundary regex build_index.py's
    chunker uses), grouping adjacent sections up to max_chars — same
    "group + cap by size" shape as clean_source.py's
    _group_adjacent/_split_range_by_size, just windowing whole sections
    instead of flagged paragraphs. A single section bigger than max_chars
    on its own is kept whole rather than truncated, same precedent as
    clean_source.py's own size-capping."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    sections = []
    if matches[0].start() > 0:
        sections.append(text[:matches[0].start()])
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[m.start():end])

    windows = []
    current = ""
    for section in sections:
        if current and len(current) + len(section) > max_chars:
            windows.append(current)
            current = section
        else:
            current += section
    if current:
        windows.append(current)
    return windows


# ── discovery ────────────────────────────────────────────────────────────────

_DISCOVERY_SYSTEM = (
    "You read passages from a Dungeons & Dragons adventure book and list "
    "every named person, named place, and named item mentioned in the passage."
)

_DISCOVERY_PROMPT = """Passage:
{window}

List every named individual (a person with a proper name — not "the guard" \
or "a villager", but an actual name like "Ireena Kolyana"), every named \
place (a specific location with a proper name — not "the forest" but \
"Svalich Woods"), AND every named item or artifact (a specific object with \
a proper name, like "the Sunsword" or "Ring of Winter Wishing" — not \
generic gear like "a longsword" or "a healing potion") mentioned anywhere \
in the passage above.

Output one line per entity, in the form:
PERSON: <name>
or
PLACE: <name>
or
ITEM: <name>

Use the name exactly as it appears in the text. If the same name appears \
multiple times, list it only once. If nothing qualifies, output nothing. \
Do not include monsters/creatures referred to only by species (e.g. "a \
giant spider"), only individually named beings, places, and items. Output \
nothing else — no explanation, no numbering."""

_DISCOVERY_LINE_RE = re.compile(r'^\s*(PERSON|PLACE|ITEM)\s*:\s*(.+?)\s*$', re.MULTILINE)
_DISCOVERY_KIND = {"PERSON": "npc", "PLACE": "location", "ITEM": "item"}


def discover_candidates(windows: list[str], model: str, ollama_url: str) -> dict[str, set[str]]:
    """Returns {"npc": {names...}, "location": {names...}, "item": {names...}}
    — union across all windows, not yet deduped/canonicalized (that's
    canonicalize()'s job)."""
    from backend.llm import ollama_chat
    from langchain_core.messages import HumanMessage, SystemMessage

    from tqdm import tqdm

    llm = ollama_chat(model=model, base_url=ollama_url, timeout=None)
    found: dict[str, set[str]] = {"npc": set(), "location": set(), "item": set()}

    for window in tqdm(windows, desc="  Discovery windows", unit="window", dynamic_ncols=True):
        response = llm.invoke([
            SystemMessage(content=_DISCOVERY_SYSTEM),
            HumanMessage(content=_DISCOVERY_PROMPT.format(window=window)),
        ])
        for m in _DISCOVERY_LINE_RE.finditer(response.content):
            kind = _DISCOVERY_KIND[m.group(1)]
            name = m.group(2).strip()
            # Guard against single-letter/very-short spurious matches (e.g. a
            # stray "G." abbreviation misread as a name) — these would each
            # match every occurrence of that substring in collate_references,
            # flooding an entity's references with garbage.
            if len(name) >= 3:
                found[kind].add(name)

    return found


# ── canonicalization ─────────────────────────────────────────────────────────

_CANON_SYSTEM = (
    "You merge a list of names into canonical entities, grouping aliases "
    "and variant spellings/forms of the same person, place, or item together."
)

_CANON_PROMPT = """Names found in a D&D adventure book (one {kind} per line):
{numbered}

Some of these may be different forms of the SAME {kind} (e.g. "Rose", \
"Rosavalda", and 'Rosavalda "Rose" Durst' are all one person). Group them.

Output one line per DISTINCT {kind}, in the form:
CANONICAL: <best full name> | ALIASES: <comma-separated other forms, or "none">

Use the fullest/most formal name as canonical when one is clearly more \
complete. Output exactly one line per distinct entity, nothing else."""

_CANON_LINE_RE = re.compile(r'^\s*CANONICAL:\s*(.+?)\s*\|\s*ALIASES:\s*(.+?)\s*$', re.MULTILINE)


CANONICALIZE_BATCH_SIZE = 150


def canonicalize(names: set[str], kind: str, model: str, ollama_url: str) -> dict[str, list[str]]:
    """Returns {canonical_name: [aliases...]}. Skip-on-doubt: any name the
    LLM response doesn't account for becomes its own canonical entity with
    no aliases — worse for dedup, but never silently drops an entity.

    Batched at CANONICALIZE_BATCH_SIZE names/call rather than one call over
    the whole set — confirmed live against the DMG's magic-item tables (a
    single-call canonicalize with no output cap and temperature=0) ran for
    5+ hours with zero progress visibility before being killed. Batches are
    sliced off the SORTED name list so alias variants of the same name
    (which usually share a prefix, e.g. "Rose"/"Rosavalda") tend to land in
    the same or an adjacent batch; canonicalization across a batch boundary
    is not attempted — same skip-on-doubt tradeoff as always, worse dedup
    but no silent drops."""
    if not names:
        return {}

    from backend.llm import ollama_chat
    from langchain_core.messages import HumanMessage, SystemMessage
    from tqdm import tqdm

    sorted_names = sorted(names)
    batches = [sorted_names[i:i + CANONICALIZE_BATCH_SIZE] for i in range(0, len(sorted_names), CANONICALIZE_BATCH_SIZE)]
    llm = ollama_chat(model=model, base_url=ollama_url, timeout=None)

    result: dict[str, list[str]] = {}
    desc = f"  Canonicalizing {kind}"
    for batch in (tqdm(batches, desc=desc, unit="batch", dynamic_ncols=True) if len(batches) > 1 else batches):
        numbered = "\n".join(batch)
        response = llm.invoke([
            SystemMessage(content=_CANON_SYSTEM),
            HumanMessage(content=_CANON_PROMPT.format(kind=kind, numbered=numbered)),
        ])

        batch_seen: set[str] = set()
        for m in _CANON_LINE_RE.finditer(response.content):
            canonical = m.group(1).strip()
            aliases_raw = m.group(2).strip()
            aliases = [] if aliases_raw.lower() == "none" else [a.strip() for a in aliases_raw.split(",")]
            result[canonical] = aliases
            batch_seen.add(canonical)
            batch_seen.update(aliases)

        for name in batch:
            if name not in batch_seen:
                result[name] = []

    return result


# ── reference collation ─────────────────────────────────────────────────────

def collate_references(
    canonical: str, aliases: list[str], book_text: str,
    context_chars: int = REFERENCE_CONTEXT_CHARS, limit: int = MAX_REFERENCES_PER_NAME,
) -> str:
    """Gather every literal mention of `canonical` or any of its `aliases`
    anywhere in the book, each with `context_chars` of surrounding text —
    this is the step that catches evidence scattered across the whole
    document rather than confined to one scene. Standalone regex scan, no
    RulesStore/Chroma dependency (same reasoning as the module docstring)."""
    windows: list[str] = []
    seen_spans: list[tuple[int, int]] = []
    for name in [canonical] + aliases:
        for m in re.finditer(re.escape(name), book_text, re.IGNORECASE):
            start = max(0, m.start() - context_chars)
            end = min(len(book_text), m.end() + context_chars)
            if any(not (end <= s or start >= e) for s, e in seen_spans):
                continue  # overlaps a window already captured
            seen_spans.append((start, end))
            windows.append(book_text[start:end])
            if len(windows) >= limit:
                return "\n\n---\n\n".join(windows)
    return "\n\n---\n\n".join(windows)


# ── profile generation ───────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)


def _generate_json_profile(system: str, prompt: str, model: str, ollama_url: str) -> dict:
    """Shared LLM-call-and-parse helper for any extractor's generate_profile.
    Skip-on-doubt: a malformed response yields an empty profile (the entity
    still gets a registry entry with aliases/type, just no synthesized
    fields) rather than aborting the whole run."""
    from backend.llm import ollama_chat
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ollama_chat(model=model, base_url=ollama_url, timeout=None)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    cleaned = _JSON_FENCE_RE.sub("", response.content).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("      ! malformed JSON response, skipping profile fields", file=sys.stderr)
        return {}


class EntityExtractor(ABC):
    """One implementation per entity type. The shared orchestration in
    extract_book() depends only on this interface, never on a specific
    type's downstream consumer (backend/tools/npc.py, world.py, ...) — a
    field rename in one tool file doesn't ripple into shared pipeline code.
    Optional per-type capabilities (like Location's generate_connections)
    are NOT part of this base interface — interface segregation, not every
    extractor needs a relationship-graph step."""

    kind: str  # "npc" | "location" | "item" — must match discover_candidates()'s
               # keys, EXCEPT "monster", which uses its own deterministic
               # header-based candidate path (discover_monster_candidates)
               # instead of discover_candidates/canonicalize — see extract_book().

    @abstractmethod
    def generate_profile(self, name: str, references: str, model: str, ollama_url: str) -> dict:
        """Return the fields this entity type's consumer tool expects."""
        ...


class NPCExtractor(EntityExtractor):
    kind = "npc"

    _SYSTEM = (
        "You extract a grounded NPC profile from passages of a D&D adventure "
        "book. Use only what the text supports — never invent traits, "
        "history, or relationships the passages don't contain."
    )
    _PROMPT = """The following are all the passages mentioning "{name}" in \
this adventure book:

{references}

Extract a profile for {name} as JSON with these fields:
{{
  "race": "",
  "occupation": "",
  "attitude": "",
  "personality_traits": [],
  "motivations": [],
  "secrets": [],
  "notes": ""
}}

personality_traits/motivations/secrets are short phrases (2-5 words each), \
not paragraphs — e.g. "gaunt and haunted", "protecting his children at any \
cost". Leave a field empty rather than padding it with a generic guess if \
the text doesn't support it. attitude should reflect how this character \
would feel about a party of adventurers encountering them, given the text \
(e.g. "hostile", "cautious", "indifferent"). Output ONLY the JSON object, \
nothing else."""

    def generate_profile(self, name: str, references: str, model: str, ollama_url: str) -> dict:
        return _generate_json_profile(
            self._SYSTEM, self._PROMPT.format(name=name, references=references), model, ollama_url,
        )


class LocationExtractor(EntityExtractor):
    kind = "location"

    _SYSTEM = (
        "You extract a grounded location profile from passages of a D&D "
        "adventure book. Use only what the text supports — never invent "
        "detail the passages don't contain."
    )
    _PROMPT = """The following are all the passages mentioning "{name}" in \
this adventure book:

{references}

Extract a profile for {name} as JSON with these fields:
{{
  "description": "",
  "area_type": "",
  "scale": "",
  "size": "",
  "notes": "",
  "points_of_interest": [],
  "hidden_elements": []
}}

description is 1-2 sentences of specific, concrete detail from the text \
(what it looks like, sounds like, any hazards/wards) — never generic, \
never blank if the text gives any detail. area_type is "indoor" or \
"outdoor". scale is "region", "settlement", "site", or "room" — pick \
whichever best matches how the text treats this place. points_of_interest \
and hidden_elements are short phrases pulled from any numbered \
site/terrain breakdown in the text (include area numbers if present, e.g. \
"Area 11 — the slave pen"). Output ONLY the JSON object, nothing else."""

    def generate_profile(self, name: str, references: str, model: str, ollama_url: str) -> dict:
        return _generate_json_profile(
            self._SYSTEM, self._PROMPT.format(name=name, references=references), model, ollama_url,
        )

    def generate_connections(
        self, locations: list[str], windows: list[str], model: str, ollama_url: str
    ) -> list[dict]:
        """Relationship-graph extraction — structurally different from a
        flat per-entity profile, so it's a separate method rather than a
        field on the profile, and not part of the base EntityExtractor
        interface other types must implement (interface segregation)."""
        if not locations:
            return []

        from backend.llm import ollama_chat
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ollama_chat(model=model, base_url=ollama_url, timeout=None)
        known = ", ".join(locations)
        connections: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()

        for window in windows:
            prompt = f"""Known named locations in this adventure: {known}

Passage:
{window}

Which pairs of the KNOWN locations above does this passage describe as \
connected, adjacent, or reachable from one another (e.g. "a road leads \
from X to Y", "Y is inside X")? Output one line per connection:
FROM: <location> | TO: <location> | VIA: <short description, e.g. "a \
muddy road, half a day's travel">

Only use location names from the known list above, exactly as given. If \
none are described as connected in this passage, output nothing."""
            response = llm.invoke([
                SystemMessage(content=self._SYSTEM), HumanMessage(content=prompt),
            ])
            for m in re.finditer(
                r'^\s*FROM:\s*(.+?)\s*\|\s*TO:\s*(.+?)\s*\|\s*VIA:\s*(.+?)\s*$',
                response.content, re.MULTILINE,
            ):
                frm, to, via = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                pair = tuple(sorted([frm, to]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                connections.append({"from": frm, "to": to, "via": via})

        return connections


class ItemExtractor(EntityExtractor):
    kind = "item"

    _SYSTEM = (
        "You extract a grounded item profile from passages of a D&D "
        "adventure book. Use only what the text supports — never invent "
        "detail the passages don't contain."
    )
    _PROMPT = """The following are all the passages mentioning "{name}" in \
this adventure book:

{references}

Extract a profile for {name} as JSON with these fields:
{{
  "item_type": "",
  "description": "",
  "rarity": "",
  "magical": false,
  "requires_attunement": false,
  "found_at": "",
  "owned_by": "",
  "notes": ""
}}

item_type is one of "weapon", "armor", "wondrous", "consumable", "quest", \
or "misc" — pick whichever best matches how the text treats this item. \
rarity is one of "common", "uncommon", "rare", "very rare", "legendary", \
"artifact", or "" if the text doesn't state one (mundane items should be \
""). description is 1-2 sentences of specific, concrete detail from the \
text (appearance, powers, history) — never generic, never blank if the \
text gives any detail. found_at is the name of the location where this \
item is found or kept, if the text states one, else "". owned_by is the \
name of the NPC who currently owns or carries this item, if the text \
states one, else "". Output ONLY the JSON object, nothing else."""

    def generate_profile(self, name: str, references: str, model: str, ollama_url: str) -> dict:
        return _generate_json_profile(
            self._SYSTEM, self._PROMPT.format(name=name, references=references), model, ollama_url,
        )


class MonsterExtractor(EntityExtractor):
    kind = "monster"

    _SYSTEM = (
        "You transcribe a D&D monster stat block into structured JSON. "
        "Copy the printed numbers exactly — never round, invent, estimate, "
        "or 'correct' anything the text doesn't literally state."
    )
    _PROMPT = """The following is a monster stat block from a D&D rulebook:

{references}

Extract this monster's stats as JSON with these fields:
{{
  "ac": 0,
  "hp": 0,
  "hit_dice": "",
  "speed": "",
  "str": 10, "dex": 10, "con": 10, "int": 10, "wis": 10, "cha": 10,
  "saving_throws": "",
  "skills": "",
  "senses": "",
  "languages": "",
  "challenge_rating": "",
  "attacks": [
    {{"name": "", "to_hit_bonus": 0, "damage_dice": "", "damage_type": "", "range_ft": ""}}
  ],
  "notes": ""
}}

ac/hp are the printed Armor Class and (average) Hit Points as integers. \
hit_dice is the printed dice formula (e.g. "6d8+12"). speed is the printed \
speed line verbatim (e.g. "30 ft., fly 60 ft."). str/dex/con/int/wis/cha \
are the six printed ability scores. saving_throws/skills/senses/languages \
are the printed lines verbatim, or "" if the block has none. \
challenge_rating is the printed CR (e.g. "5" or "1/2"). attacks is one \
entry per printed attack action, with to_hit_bonus as the printed "+N" \
integer and damage_dice as the printed dice (e.g. "2d6+4") — omit an attack \
entirely rather than guessing a field you can't find. Output ONLY the JSON \
object, nothing else."""

    def generate_profile(self, name: str, references: str, model: str, ollama_url: str) -> dict:
        return _generate_json_profile(
            self._SYSTEM, self._PROMPT.format(references=references), model, ollama_url,
        )


EXTRACTORS: dict[str, EntityExtractor] = {
    "npc": NPCExtractor(), "location": LocationExtractor(), "item": ItemExtractor(),
    "monster": MonsterExtractor(),
}


# ── orchestration ────────────────────────────────────────────────────────────

def _partial_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".partial")


def _out_path(book: str, output_dir: str, source_type: str) -> Path:
    # Adventures use a per-book subfolder convention already
    # (docs/source/adventures/{slug}/_entities.json). Core rulebooks are flat
    # files with no subfolder — write to a dedicated _entities/ subfolder
    # under the core input dir instead of littering it with stray files.
    if source_type == "core":
        return Path(f"{output_dir}/_entities/{book}.json")
    return Path(f"{output_dir}/{book}/_entities.json")


_ALL_KINDS = ("npc", "location", "item", "monster")


async def extract_book(
    book: str, input_dir: str, output_dir: str, model: str, ollama_url: str,
    dry_run: bool = False, force: bool = False, source_type: str = "adventure",
    kinds: tuple[str, ...] = _ALL_KINDS, write_postgres: bool = False,
) -> None:
    out_path = _out_path(book, output_dir, source_type)

    # Resume-aware: `_checkpoint()` below atomically renames .partial ->
    # out_path after EVERY entity, so out_path existing does NOT mean the
    # whole book finished — only registry.get("_complete") does. A prior bug
    # here treated any existing out_path as "fully done," silently skipping
    # entities that were never actually processed after a kill mid-run.
    registry: dict = {}
    if out_path.exists():
        existing_registry = json.loads(out_path.read_text(encoding="utf-8"))
        if existing_registry.get("_complete") and not force:
            print(f"SKIP  {book}  (already done — use --force to redo)")
            return
        if not force:
            registry = existing_registry
            done = sorted(k for k in registry if not k.startswith("_"))
            print(f"  Resuming {book} — {len(done)} entit(y/ies) already done")

    print(f"── {book} ({source_type}, kinds={','.join(kinds)}) ──")
    book_text = _read_book_text(book, input_dir, source_type)
    if not book_text:
        print(f"  No source text found for {book!r} ({source_type})", file=sys.stderr)
        return

    # A pure stat-block book (e.g. the Monster Manual) has essentially no
    # named NPCs/locations/items — running the windowed discovery LLM pass
    # for those kinds anyway would burn many calls discovering empty sets.
    # --kinds lets a core-book run skip straight to whichever kinds actually
    # apply to that book, which matters given these passes can run for hours.
    discovery_kinds = [k for k in ("npc", "location", "item") if k in kinds]

    windows = _split_into_windows(book_text)
    print(f"  {len(windows)} discovery window(s)")

    if discovery_kinds:
        print("  Discovering candidates...")
        candidates = {k: v for k, v in discover_candidates(windows, model, ollama_url).items() if k in discovery_kinds}
    else:
        candidates = {}
    print(f"    npc={len(candidates.get('npc', ()))} location={len(candidates.get('location', ()))} "
          f"item={len(candidates.get('item', ()))} candidate name(s)")

    print("  Canonicalizing...")
    canonical = {
        kind: canonicalize(names, kind, model, ollama_url)
        for kind, names in candidates.items()
    }
    for kind, entries in canonical.items():
        print(f"    {kind}: {len(entries)} canonical entit(y/ies)")

    # Monsters skip discovery/canonicalization entirely — see
    # discover_monster_candidates's docstring. Deterministic, no LLM call.
    monster_candidates = discover_monster_candidates(book_text) if "monster" in kinds else {}
    print(f"    monster: {len(monster_candidates)} stat-block candidate(s) (deterministic, no LLM call)")

    if dry_run:
        for kind, entries in canonical.items():
            for name, aliases in entries.items():
                suffix = f"  (aka {', '.join(aliases)})" if aliases else ""
                print(f"    [{kind}] {name}{suffix}")
        for name in monster_candidates:
            print(f"    [monster] {name}")
        return

    partial = _partial_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint():
        partial.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        partial.replace(out_path)

    lore_store = None
    if write_postgres:
        from sqlalchemy.ext.asyncio import create_async_engine
        from backend.config import settings
        from backend.stores.lore_store import LoreStore
        lore_store = LoreStore(create_async_engine(settings.database_url))

    async def _write_postgres(kind: str, name: str, aliases: list[str], profile: dict) -> None:
        if lore_store is None:
            return
        try:
            await lore_store.upsert_entity(
                book_slug=book, entity_type=kind, canonical_name=name, profile=profile,
                aliases=aliases, spoiler_tier=profile.get("spoiler_tier", "public"),
                source_type=source_type,
            )
        except Exception as e:
            print(f"      ! Postgres write failed ({e}) — JSON checkpoint is still intact", file=sys.stderr)

    from tqdm import tqdm

    all_entries = [
        (kind, name, aliases)
        for kind, entries in canonical.items()
        for name, aliases in entries.items()
    ]
    pbar = tqdm(all_entries, desc="  Extracting entities", unit="entity", dynamic_ncols=True)
    for kind, name, aliases in pbar:
        if name in registry:  # already done in a prior (interrupted) run
            pbar.set_postfix_str(f"[{kind}] {name} (skipped, already done)"[:60])
            continue
        pbar.set_postfix_str(f"[{kind}] {name}"[:60])
        extractor = EXTRACTORS[kind]
        references = collate_references(name, aliases, book_text, **REFERENCE_PARAMS[kind])
        if not references:
            tqdm.write(f"      ! no references found for [{kind}] {name}, skipping")
            continue
        try:
            profile = extractor.generate_profile(name, references, model, ollama_url)
        except Exception as e:
            tqdm.write(f"      ! generation failed for [{kind}] {name} ({e}), skipping")
            continue
        registry[name] = {"type": kind, "aliases": aliases, "profile": profile}
        _checkpoint()
        await _write_postgres(kind, name, aliases, profile)

    if "monster" in kinds:
        monster_extractor = EXTRACTORS["monster"]
        mbar = tqdm(list(monster_candidates.items()), desc="  Extracting monsters", unit="monster", dynamic_ncols=True)
        for name, section_text in mbar:
            if name in registry:
                mbar.set_postfix_str(f"{name} (skipped, already done)"[:60])
                continue
            mbar.set_postfix_str(name[:60])
            try:
                profile = monster_extractor.generate_profile(name, section_text, model, ollama_url)
            except Exception as e:
                tqdm.write(f"      ! generation failed for [monster] {name} ({e}), skipping")
                continue
            registry[name] = {"type": "monster", "aliases": [], "profile": profile}
            _checkpoint()
            await _write_postgres("monster", name, [], profile)

    location_extractor = EXTRACTORS["location"]
    if isinstance(location_extractor, LocationExtractor) and canonical.get("location") and "_connections" not in registry:
        print("  Generating location connections...")
        location_names = list(canonical["location"].keys())
        connections = location_extractor.generate_connections(location_names, windows, model, ollama_url)
        registry["_connections"] = connections
        _checkpoint()
        print(f"    {len(connections)} connection(s)")

    registry["_complete"] = True
    _checkpoint()
    print(f"  Wrote {out_path}")


DEFAULT_CORE_INPUT = "docs/source/core"


async def _main_async() -> None:
    ap = argparse.ArgumentParser(description="Precompute an NPC/Location/Item/Monster entity index for a book.")
    ap.add_argument("--input",       default=None, help="folder to read from (default: docs/source/adventures or docs/source/core, per --source-type)")
    ap.add_argument("--output",      default=None, help="folder to write entity index under (default: same as --input)")
    ap.add_argument("--book",        default=None, help="process a single book (adventure slug / core rulebook filename stem)")
    ap.add_argument("--source-type", default="adventure", choices=["adventure", "core"],
                    help="'adventure' (default): one subfolder per book under --input. "
                         "'core': one flat .md file per book directly under --input, "
                         "same convention as build_index.py's --source-type.")
    ap.add_argument("--model",      default=DEFAULT_MODEL, help="Ollama model to use")
    ap.add_argument("--ollama-url", default=None, help="Ollama base URL")
    ap.add_argument("--dry-run",    action="store_true", help="report canonical entities without generating profiles")
    ap.add_argument("--force",      action="store_true", help="re-process even if the entity index already exists")
    ap.add_argument("--kinds",      default=",".join(_ALL_KINDS),
                    help=f"comma-separated entity kinds to extract (default: all of {','.join(_ALL_KINDS)}). "
                         "E.g. --kinds monster for a pure stat-block book like the Monster Manual, "
                         "skipping the (mostly empty) npc/location/item discovery pass entirely.")
    ap.add_argument("--write-postgres", action="store_true",
                    help="also upsert each entity into the Postgres Lore Registry "
                         "(lore_entities/lore_entity_aliases) as it's generated, "
                         "not just the JSON debug artifact.")
    args = ap.parse_args()

    ollama_url = args.ollama_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA
    input_dir = args.input or (DEFAULT_CORE_INPUT if args.source_type == "core" else DEFAULT_INPUT)
    output_dir = args.output or input_dir
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())

    if args.book:
        books = [args.book]
    elif args.source_type == "core":
        books = sorted(p.stem for p in Path(input_dir).glob("*.md"))
    else:
        books = sorted(p.name for p in Path(input_dir).iterdir() if p.is_dir())

    if not books:
        print(f"No books found in {input_dir}/ (source_type={args.source_type})")
        return

    for book in books:
        await extract_book(
            book, input_dir, output_dir, args.model, ollama_url, args.dry_run, args.force,
            args.source_type, kinds, args.write_postgres,
        )

    print("\nDone.")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
