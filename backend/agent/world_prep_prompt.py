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
8. When you've covered the material above, reply with a short plain-text
   summary of what you created. Do not narrate or roleplay — this is a
   data-entry pass, not a scene."""
