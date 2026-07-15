from backend.models import Campaign


def get_world_prep_prompt(campaign: Campaign, book: str, seed_context: str) -> str:
    return f"""You are pre-populating the region-scale world map for the campaign
"{campaign.name}" before play begins, based on the adventure "{book}".

Below is retrieved text from the adventure covering its regional overview,
named settlements/landmarks, and any stated travel distances or times.

{seed_context}

Instructions:
1. For every named REGION-SCALE place the party might travel to (town, city,
   dungeon-as-a-whole, notable landmark), call create_location(name=..., scale="region", ...).
   Skip site-scale detail (individual rooms) — that's for play, not prep.
2. Whenever the text states or clearly implies a distance or travel time
   between two such places, call connect_locations with distance_miles and
   terrain. If only a travel time is given (e.g. "two days' ride"), convert
   using 24 miles/day at normal pace and say so in the notes field.
3. Do NOT invent a distance that isn't stated or clearly implied by the text.
   It's fine to create a location with no connections — gaps can be filled
   in later during play using the same tools.
4. area_type for a region-scale place is almost always "outdoor" (use
   "underground" only for a subterranean city/dungeon) — it is NOT the
   settlement type, so never pass values like "city" or "town" there.
5. Leave is_visible at its default (true) for every connection you create —
   a road between two named settlements in the adventure is common knowledge,
   not something hidden from the party. Do not set it to false.
6. Omit an argument entirely rather than passing null/None for it — every
   optional field already has a sensible default.
7. You may call search_rules for more detail on any location you're unsure about.
8. For each settlement/notable-area location you create, if the text gives
   enough to ground a rough layout (named streets, districts, notable
   buildings, terrain), also call set_location_grid with a small ASCII grid
   (5-ft squares) — this is what lets players browse "what does this town
   look like" before ever setting foot there. This is OPTIONAL and
   best-effort: skip it entirely for a location the text doesn't describe
   physically — DO NOT invent a street layout that isn't grounded in the
   source text. A location with no grid just gets authored live later,
   during play.
9. When you've covered the material above, reply with a short plain-text
   summary of what you created. Do not narrate or roleplay — this is a
   data-entry pass, not a scene."""


def get_npc_prep_prompt(campaign: Campaign, book: str, opening_location: str, context: str) -> str:
    """NPC roster only — deliberately does NOT also ask for the location
    detail call (see get_opening_location_prompt for that, invoked as a
    SEPARATE agent run). Verified live: combining both in one long
    tool-calling sequence let the location call get starved — the model ran
    out of budget partway through a long roster and never reached it. Two
    separate, focused invocations each get their own dedicated budget."""
    return f"""You are pre-populating the OPENING SCENE's cast of characters for
the campaign "{campaign.name}", based on the adventure "{book}" — specifically
the part of the book covering "{opening_location}", where play actually begins.

This is the single most important scene in this entire prep pass. Nothing
else will be prepared before the players see it — every named character the
party encounters in their very first minutes of play depends on what you do
here. A generic, invented opening (a couple of vague "mysterious" characters)
is exactly the failure this pass exists to prevent.

Below is the full text covering "{opening_location}" from the adventure:

{context}

Instructions:
1. FIRST, before calling any tool, write out a plain numbered list (in your own
   reply text, not a tool call) of EVERY named individual physically present
   at or tied to this opening scene — not just the obvious protagonist-
   adjacent ones. This includes any commander/antagonist/authority figure
   running the place, AND every named companion/prisoner/bystander described,
   even ones who only get a single sentence. Do not skip minor-seeming names,
   and do not invent names that aren't in the text above. This list is your
   checklist for step 2 — commit to it now, a fixed target count, not
   something you discover as you go.
2. For each name on your list, call lookup_entity(name) FIRST — if the canon
   Lore Registry already has a grounded profile for this book, use those
   fields directly in create_npc rather than re-deriving them yourself from
   the raw text below. If lookup_entity finds nothing, derive the profile
   from the text as usual. Then call create_npc once per name on YOUR list
   from step 1, in order, until every single one has a real record —
   completing the WHOLE roster matters more than how much you write for any
   one individual. Keep each
   field to short phrases, not paragraphs — this is data entry, not prose.
   `personality_traits`/`motivations` are not optional if the text gives this
   individual any characterization at all (a want, a fear, a habit), but keep
   each entry to 2-5 words, e.g. `personality_traits=["devil-may-care",
   "gambling obsession"]`, `motivations=["chasing coin and a good wager"]` —
   short and grounded, not an empty list AND not a paragraph. Skip these
   fields only for a true one-line mention with nothing characterizing said.
   Ground race/occupation/attitude/secrets/notes the same way — brief,
   grounded, not padded. `attitude` should reflect how they'd actually feel
   about the party given the text (e.g. a fellow captive is likely "cautious"
   or "indifferent" at first meeting, a hostile captor is likely "hostile" or
   "suspicious").
3. You may call search_rules for anything you need more detail on, but do not
   invent facts the retrieved text doesn't support.
4. Before you write your final summary, check yourself against your own list
   from step 1: does the number of create_npc calls you actually made match
   the number of names on that list? If not, you are NOT done — keep calling
   create_npc until it does. Completeness across the whole roster matters
   more than extra polish on the first few individuals.
5. Only once that's confirmed true, reply with a short plain-text summary of
   what you created. Do not narrate or roleplay — this is a data-entry pass,
   not a scene."""


def get_opening_location_prompt(
    campaign: Campaign, book: str, opening_location: str, context: str, npc_names: list[str]
) -> str:
    """Site-detail-only — a separate, focused agent run from
    get_npc_prep_prompt (see its docstring for why), so this one call always
    gets its own dedicated tool-call budget regardless of roster size."""
    npc_list = ", ".join(npc_names) if npc_names else "(none created yet)"
    return f"""You are pre-populating site detail for the OPENING SCENE's location,
"{opening_location}", for the campaign "{campaign.name}" — based on the
adventure "{book}", where play actually begins.

These named individuals already exist as real records and are present in
this scene: {npc_list}.

Below is the full text covering "{opening_location}" from the adventure:

{context}

Instructions:
1. Call set_opening_location_detail exactly once for "{opening_location}".
   `description` is NOT optional — 1-2 sentences of specific, concrete detail
   from the text above (what it looks like, sounds like, any wards/hazards/
   restraints described — not a generic one-liner, and never blank).
2. `points_of_interest` and `hidden_elements` should be short phrases pulled
   from any site/terrain breakdown in the text above (numbered areas, guard
   posts, shrines, exits) — include the numbering if the text has it (e.g.
   "Area 11 — the slave pen").
3. `current_npcs` should include the named individuals listed above, plus any
   others you can confirm from the text are actually present in THIS scene
   (not anyone merely mentioned elsewhere in the book).
4. Leave `make_current` at its default.
5. You may call search_rules for anything you need more detail on, but do not
   invent facts the retrieved text doesn't support.
6. Once you've made the call, reply with a short plain-text summary of what
   you set. Do not narrate or roleplay — this is a data-entry pass, not a
   scene."""
