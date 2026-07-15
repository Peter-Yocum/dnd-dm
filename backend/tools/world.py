from langchain_core.tools import tool

from backend.models import AreaType, Location, LocationConnection, LocationScale, TravelTerrain
from backend.rag.entity_resolution import find_candidate_matches
from backend.stores.campaign_store import CampaignStore
from backend.stores.lore_store import LoreStore
from backend.tools._helpers import (
    advance_clock, apply_long_rest, apply_short_rest, find_connection, find_container, find_location,
    opposite_direction,
)


def make_movement_tools(campaign_id: str, store: CampaignStore) -> list:

    @tool
    async def get_current_location() -> str:
        """Get the party's current location — description, terrain, lighting,
        points of interest, visible exits, and present NPCs. Call at the start
        of any scene or whenever the players ask to look around."""
        campaign = await store.load(campaign_id)
        if not campaign.current_location_id:
            return "The party's location is not set. Use move_party to place them."
        loc = next((l for l in campaign.locations if l.id == campaign.current_location_id), None)
        if not loc:
            return "Current location not found in campaign data."
        lines = [
            f"=== {loc.name} ===",
            f"Type: {loc.area_type.value}  Lighting: {loc.lighting.value}",
        ]
        if loc.size:
            lines.append(f"Size: {loc.size}")
        if loc.description:
            lines.append(loc.description)
        if loc.terrain_features:
            lines.append("Terrain: " + ", ".join(loc.terrain_features))
        if loc.points_of_interest:
            lines.append("Points of interest:")
            lines += [f"  - {p}" for p in loc.points_of_interest]
        if loc.connections:
            lines.append("Exits:")
            for conn in loc.connections:
                passable = "" if conn.is_passable else " [BLOCKED]"
                visible = "" if conn.is_visible else " [HIDDEN]"
                lines.append(f"  {conn.direction} → {conn.to_location_name}{passable}{visible}")
        if loc.current_npcs:
            lines.append("Present NPCs: " + ", ".join(loc.current_npcs))
        if loc.notes:
            lines.append(f"DM notes: {loc.notes}")
        return "\n".join(lines)

    @tool
    async def move_party(location_name: str) -> str:
        """Move the party to a named location. The location must already exist
        in the campaign. Also updates the NPC presence list at the old and new
        location. Call after the players travel or enter a new area."""
        campaign = await store.load(campaign_id)
        loc = find_location(campaign, location_name)
        if not loc:
            return f"No location named '{location_name}' in the campaign."
        campaign.current_location_id = loc.id
        loc.visited = True
        await store.save(campaign)
        npc_str = (", ".join(loc.current_npcs)) if loc.current_npcs else "no one"
        return (
            f"The party moves to {loc.name}. "
            f"Lighting: {loc.lighting.value}. "
            f"Present: {npc_str}."
        )

    @tool
    async def reveal_hidden_element(location_name: str, element_index: int) -> str:
        """Reveal a hidden element in a location to the party. Use the 0-based index
        from get_current_location's DM-only hidden list. Moves the element to the
        visible points_of_interest list."""
        campaign = await store.load(campaign_id)
        loc = find_location(campaign, location_name)
        if not loc:
            return f"No location named '{location_name}' found."
        if not (0 <= element_index < len(loc.hidden_elements)):
            return f"No hidden element at index {element_index} in {loc.name}."
        element = loc.hidden_elements.pop(element_index)
        loc.points_of_interest.append(element)
        await store.save(campaign)
        return f"The party discovers: {element}"

    @tool
    async def open_container(container_name: str) -> str:
        """Open a container and reveal its contents to the party. Returns everything
        inside — items and currency. Marks the container as open in the campaign."""
        campaign = await store.load(campaign_id)

        # Also check party treasury by name
        container = None
        if campaign.party_treasury.name.lower() == container_name.lower():
            container = campaign.party_treasury
        else:
            container = find_container(campaign, container_name)

        if not container:
            return f"No container named '{container_name}' found."
        if container.is_locked:
            return f"{container.name} is locked (DC {container.lock_dc or '?'} to pick)."
        container.is_open = True
        await store.save(campaign)

        lines = [f"=== {container.name} ==="]
        if container.description:
            lines.append(container.description)
        if container.contents:
            lines.append("Items:")
            lines += [
                f"  - {i.name}" + (f" x{i.quantity}" if i.quantity > 1 else "")
                + (f" ({i.description})" if i.description else "")
                for i in container.contents
            ]
        else:
            lines.append("No items.")
        c = container.currency
        coins = []
        if c.pp: coins.append(f"{c.pp} pp")
        if c.gp: coins.append(f"{c.gp} gp")
        if c.ep: coins.append(f"{c.ep} ep")
        if c.sp: coins.append(f"{c.sp} sp")
        if c.cp: coins.append(f"{c.cp} cp")
        if coins:
            lines.append("Currency: " + ", ".join(coins))
        return "\n".join(lines)

    return [get_current_location, move_party, reveal_hidden_element, open_container]


def _parse_enum(enum_cls, value: str, field_name: str):
    """Returns (enum_value, None) or (None, error_message). Tool functions
    must not let a bad LLM-supplied enum string raise — an uncaught
    ValueError here propagates out of the whole agent run (see world-prep
    verification), not just this one tool call."""
    try:
        return enum_cls(value), None
    except ValueError:
        valid = ", ".join(e.value for e in enum_cls)
        return None, f"Invalid {field_name} '{value}' — must be one of: {valid}."


def make_authoring_tools(
    campaign_id: str,
    store: CampaignStore,
    lore_store: LoreStore | None = None,
    books_in_play: list[str] | None = None,
) -> list:

    @tool
    async def create_location(
        name: str,
        description: str | None = None,
        area_type: str = "outdoor",
        scale: str = "region",
        size: str | None = None,
        notes: str | None = None,
        force: bool = False,
    ) -> str:
        """Create a new named location in the campaign world. area_type must
        be one of: indoor, outdoor, underground, aquatic, aerial — for a
        settlement or region use 'outdoor' (or 'underground' for an
        underdark city). Use scale='region' for towns, cities,
        dungeons-as-a-whole, and other overworld-map places; use scale='site'
        for a room/chamber that needs turn-by-turn detail — rare during
        world-prep, mostly used once play begins. If a location with this
        exact name already exists, returns its existing details instead of
        creating a duplicate — call connect_locations to link it to another
        place instead. If a close-but-not-exact name match already exists (in
        this campaign or the canon Lore Registry), returns a warning instead —
        call lookup_entity to check first, or pass force=True if this is
        genuinely a different place. Lighting, terrain features, and points
        of interest are added later once play begins, not during world-prep."""
        campaign = await store.load(campaign_id)
        existing = find_location(campaign, name)
        if existing:
            return f"Location '{existing.name}' already exists (id={existing.id}, scale={existing.scale.value}). Use connect_locations to link it."
        if not force:
            existing_names = [l.name for l in campaign.locations]
            if lore_store is not None:
                existing_names += await lore_store.find_candidates(books_in_play or [], "location")
            matches = find_candidate_matches(name, existing_names)
            if matches:
                return (
                    f"'{name}' is a close match to existing location(s): {', '.join(matches)}. "
                    f"Call lookup_entity('{name}') to check first, or call create_location again "
                    f"with force=True if this is genuinely a different place."
                )
        area_type_val, err = _parse_enum(AreaType, area_type, "area_type")
        if err:
            return err
        scale_val, err = _parse_enum(LocationScale, scale, "scale")
        if err:
            return err
        loc = Location(
            name=name,
            description=description or "",
            area_type=area_type_val,
            scale=scale_val,
            size=size or "",
            notes=notes or "",
        )
        campaign.locations.append(loc)
        await store.save(campaign)
        return f"Created location '{loc.name}' (id={loc.id}, scale={loc.scale.value})."

    @tool
    async def connect_locations(
        from_location: str,
        to_location: str,
        direction: str | None = None,
        distance_ft: int | None = None,
        distance_miles: float | None = None,
        terrain: str | None = None,
        bidirectional: bool = True,
        is_visible: bool = True,
        is_passable: bool = True,
        notes: str | None = None,
    ) -> str:
        """Create or update a travel connection between two EXISTING locations
        (call create_location first for each side if needed). Provide EITHER
        distance_ft (site scale, e.g. a corridor) OR distance_miles + terrain
        (region scale, e.g. a road between towns) — whichever the source text
        specifies; leave both None if a path exists but no distance is stated
        anywhere — do not invent plausible mileage. By default the connection
        is created in both directions (bidirectional=True) since most roads
        and passages are two-way; set bidirectional=False for a genuinely
        one-way feature. Calling this again for the same pair of locations
        updates the existing connection instead of creating a duplicate.
        Leave is_visible at its default True for any well-known public route
        (a road between two named settlements is common knowledge) — only set
        it False for a route that is genuinely secret or undiscovered."""
        campaign = await store.load(campaign_id)
        src = find_location(campaign, from_location)
        dst = find_location(campaign, to_location)
        missing = [n for n, l in [(from_location, src), (to_location, dst)] if not l]
        if missing:
            return f"Unknown location(s): {', '.join(missing)}. Call create_location first."
        terrain_val = None
        if terrain:
            terrain_val, err = _parse_enum(TravelTerrain, terrain, "terrain")
            if err:
                return err
        direction = direction or ""
        notes = notes or ""

        def _upsert(a: Location, b: Location, dir_: str) -> None:
            conn = find_connection(a, b.name)
            if conn is None:
                conn = LocationConnection(to_location_id=b.id, to_location_name=b.name)
                a.connections.append(conn)
            conn.to_location_name = b.name
            if dir_:
                conn.direction = dir_
            if distance_ft is not None:
                conn.distance_ft = distance_ft
            if distance_miles is not None:
                conn.distance_miles = distance_miles
            if terrain_val:
                conn.terrain = terrain_val
            conn.is_visible = is_visible
            conn.is_passable = is_passable
            if notes:
                conn.notes = notes

        _upsert(src, dst, direction)
        if bidirectional:
            _upsert(dst, src, opposite_direction(direction))
        await store.save(campaign)
        if distance_miles is not None:
            dist = f"{distance_miles} mi" + (f" ({terrain})" if terrain else "")
        elif distance_ft is not None:
            dist = f"{distance_ft} ft"
        else:
            dist = "distance unknown"
        return f"Connected {src.name} ↔ {dst.name} ({dist}, {'bidirectional' if bidirectional else 'one-way'})."

    @tool
    async def set_location_grid(
        location_name: str,
        grid: list[str],
        legend: dict[str, str] | None = None,
    ) -> str:
        """Author (or replace) a location's grid map — ANY scale, not just a
        dungeon room: a settlement's street layout, a wilderness clearing's
        terrain, a building's interior all work the same way. Sized in 5-ft
        squares (standard D&D grid scale). Every row must be the same
        length (a rectangular grid). '.' always means open/passable ground
        and never needs a legend entry; every OTHER symbol used (walls,
        doors, trees, rocks, water, difficult terrain, crates, ...) MUST
        have a legend entry explaining what it is — invent whatever
        vocabulary the scene actually calls for, there's no fixed symbol set.

        Call this whenever entering a scene where positioning will matter —
        a room, a street, a clearing — not only right before combat (combat
        specifically REQUIRES one; see start_encounter). Skip it for scenes
        where exact positioning is irrelevant."""
        campaign = await store.load(campaign_id)
        loc = find_location(campaign, location_name)
        if not loc:
            return f"No location named '{location_name}' found. Call create_location first."
        if not grid:
            return "grid must have at least one row."
        width = len(grid[0])
        if any(len(row) != width for row in grid):
            return "Every row must be the same length (a rectangular grid) — check for a short/long row."
        legend = legend or {}
        used_symbols = {ch for row in grid for ch in row if ch != "."}
        unlegended = sorted(used_symbols - set(legend))
        if unlegended:
            return (
                f"These symbols appear in the grid but have no legend entry: {', '.join(unlegended)}. "
                f"Add a legend entry for each (e.g. legend={{'#': 'wall'}}), or use '.' for open ground."
            )
        loc.grid = list(grid)
        loc.legend = dict(legend)
        await store.save(campaign)
        return f"Grid set for '{loc.name}': {width}x{len(grid)} squares, legend: {legend or '(none — all open ground)'}."

    @tool
    async def get_location_grid(location_name: str = "") -> str:
        """Get a location's grid map (battle-map notation: columns A, B, C...,
        rows 1, 2, 3...) plus its legend, for spatial reasoning ("you're 15 ft
        from the door"). Leave location_name blank for the party's current
        location. Returns a clear message if no grid has been authored yet —
        call set_location_grid first."""
        campaign = await store.load(campaign_id)
        if location_name:
            loc = find_location(campaign, location_name)
            if not loc:
                return f"No location named '{location_name}' found."
        else:
            if not campaign.current_location_id:
                return "The party's current location is not set."
            loc = next((l for l in campaign.locations if l.id == campaign.current_location_id), None)
            if not loc:
                return "Current location not found in campaign data."
        if not loc.grid:
            return f"No grid authored for '{loc.name}' yet — call set_location_grid first."
        col_letters = "   " + " ".join(chr(65 + i) for i in range(len(loc.grid[0])))
        lines = [f"=== {loc.name} grid ({len(loc.grid[0])}x{len(loc.grid)} squares, 5 ft each) ===", col_letters]
        for i, row in enumerate(loc.grid, 1):
            lines.append(f"{i:>2} " + " ".join(row))
        if loc.legend:
            lines.append("Legend: " + ", ".join(f"{sym}={meaning}" for sym, meaning in loc.legend.items()))
        lines.append("Legend: .=open/passable ground")
        return "\n".join(lines)

    return [create_location, connect_locations, set_location_grid, get_location_grid]


_PACE_MI_PER_DAY = {"slow": 18, "normal": 24, "fast": 30}


def make_travel_tools(campaign_id: str, store: CampaignStore) -> list:

    @tool
    async def get_travel_estimate(destination: str) -> str:
        """Estimate travel time to a named destination via a DIRECT known
        route from the party's current location, at slow/normal/fast pace
        (18/24/30 miles/day per the DMG). Only works for a direct connection —
        no multi-hop routing yet; if none exists, say so rather than guessing,
        the party may need to travel via an intermediate location first."""
        campaign = await store.load(campaign_id)
        if not campaign.current_location_id:
            return "The party's current location is not set."
        cur = next((l for l in campaign.locations if l.id == campaign.current_location_id), None)
        if not cur:
            return "Current location not found in campaign data."
        conn = find_connection(cur, destination)
        if not conn:
            return f"No direct known route from {cur.name} to '{destination}'."
        if not conn.is_visible:
            return f"The party doesn't know of a route from {cur.name} to '{destination}'."
        if conn.distance_miles is None:
            return f"A route to {conn.to_location_name} is known but no distance is recorded for it yet."
        if not conn.is_passable:
            return f"The route to {conn.to_location_name} is currently blocked. {conn.notes}"
        lines = [
            f"{cur.name} → {conn.to_location_name}: {conn.distance_miles} mi"
            + (f" via {conn.terrain.value}" if conn.terrain else "")
        ]
        for pace, mi in _PACE_MI_PER_DAY.items():
            days = conn.distance_miles / mi
            lines.append(f"  {pace}: {days:.1f} days ({mi} mi/day)")
        return "\n".join(lines)

    @tool
    async def travel_to(destination: str, pace: str = "normal") -> str:
        """Move the party to a named destination via a direct known
        region-scale route, advancing days_elapsed and time_of_day by the
        computed travel duration. pace is 'slow'/'normal'/'fast'. Use this
        instead of move_party whenever meaningful game time passes; use
        plain move_party for same-scene moves. Fails if no direct route with
        a recorded distance exists — call get_travel_estimate first."""
        if pace not in _PACE_MI_PER_DAY:
            return "pace must be 'slow', 'normal', or 'fast'."
        campaign = await store.load(campaign_id)
        if not campaign.current_location_id:
            return "The party's current location is not set."
        cur = next((l for l in campaign.locations if l.id == campaign.current_location_id), None)
        if not cur:
            return "Current location not found in campaign data."
        conn = find_connection(cur, destination)
        if not conn:
            return f"No direct known route from {cur.name} to '{destination}'."
        if conn.distance_miles is None:
            return (
                f"No recorded distance to {conn.to_location_name} — cannot advance the clock. "
                "Use move_party for a same-scene move, or record a distance with connect_locations first."
            )
        if not conn.is_passable:
            return f"The route to {conn.to_location_name} is blocked: {conn.notes}"
        dest = next(l for l in campaign.locations if l.id == conn.to_location_id)
        hours = (conn.distance_miles / _PACE_MI_PER_DAY[pace]) * 24
        days_advanced = advance_clock(campaign, hours)
        campaign.current_location_id = dest.id
        dest.visited = True
        await store.save(campaign)
        npc_str = ", ".join(dest.current_npcs) if dest.current_npcs else "no one"
        return (
            f"The party travels {conn.distance_miles} mi to {dest.name} at {pace} pace "
            f"(~{hours:.0f} hours). {days_advanced} day(s) pass. "
            f"Now day {campaign.days_elapsed}, {campaign.time_of_day.value}. Present: {npc_str}."
        )

    @tool
    async def advance_time(hours: float, reason: str) -> str:
        """Advance the campaign clock (days_elapsed/time_of_day) by `hours`
        of in-fiction time for a narrated time-skip that ISN'T travel — a
        stakeout, an evening spent in town, a long conversation that runs
        past dusk, "a week passes" narration, etc. Use travel_to instead for
        actual region-scale travel between named locations (it advances the
        clock itself from distance/pace). See the Time passage prompt
        section for how much to estimate. `reason` is a short label for the
        resolution report (e.g. "the party keeps watch overnight"), not
        shown to the player directly."""
        if hours <= 0:
            return "hours must be positive."
        campaign = await store.load(campaign_id)
        days_advanced = advance_clock(campaign, hours)
        await store.save(campaign)
        return (
            f"{reason}: ~{hours:.1f} hour(s) pass. {days_advanced} day(s) advanced. "
            f"Now day {campaign.days_elapsed}, {campaign.time_of_day.value}."
        )

    @tool
    async def take_rest(kind: str) -> str:
        """Apply a whole-party short or long rest when the party actually
        rests/sleeps/makes camp as part of the narrative (not just a player
        clicking the UI rest buttons) — kind is 'short' or 'long'. Same
        deterministic effects as the UI buttons (HP/spell slots/hit dice/
        exhaustion, clock advances 1 hour for short or 8 for long); this is
        just the in-fiction call path for when resting happens as part of
        the story rather than a manual click."""
        if kind not in ("short", "long"):
            return "kind must be 'short' or 'long'."
        campaign = await store.load(campaign_id)
        summary = apply_long_rest(campaign) if kind == "long" else apply_short_rest(campaign)
        await store.save(campaign)
        return summary

    return [get_travel_estimate, travel_to, advance_time, take_rest]


def make_opening_detail_tools(campaign_id: str, store: CampaignStore) -> list:
    """Prep-only tool for the one-shot world-prep opening-scene seeding pass
    (backend/agent/world_prep.py) — deliberately NOT included in make_tools()'s
    live aggregate. create_location's own docstring defers site-scale detail
    (points_of_interest/hidden_elements) to live play for every OTHER
    location; this is the one deliberate, scoped exception, for the single
    location where play actually begins, where a rich upfront description
    grounded in the book directly prevents the DM improvising the very first
    scene from nothing."""

    @tool
    async def set_opening_location_detail(
        location_name: str,
        description: str | None = None,
        area_type: str = "outdoor",
        scale: str = "region",
        points_of_interest: list[str] | None = None,
        hidden_elements: list[str] | None = None,
        current_npcs: list[str] | None = None,
        make_current: bool = True,
    ) -> str:
        """Set rich site detail on the opening scene's location — where play
        actually begins — grounded in the adventure text. Finds the location
        by name (falling back to a substring match in either direction, since
        an earlier region-scale seeding pass may have created it under a
        slightly different name, e.g. "Velkynvelve Outpost" vs "Velkynvelve"),
        or creates it fresh if it doesn't exist yet. current_npcs should list
        only who's actually present in THIS scene, not everyone mentioned
        anywhere in the book. Sets this as the party's current location
        (make_current=True, the default) so session 1 doesn't depend on the
        live model separately calling move_party correctly."""
        campaign = await store.load(campaign_id)
        loc = find_location(campaign, location_name)
        if not loc:
            n = location_name.lower()
            loc = next(
                (l for l in campaign.locations if n in l.name.lower() or l.name.lower() in n),
                None,
            )
        if not loc:
            area_type_val, err = _parse_enum(AreaType, area_type, "area_type")
            if err:
                return err
            scale_val, err = _parse_enum(LocationScale, scale, "scale")
            if err:
                return err
            loc = Location(name=location_name, area_type=area_type_val, scale=scale_val)
            campaign.locations.append(loc)
        if description is not None:
            loc.description = description
        if points_of_interest:
            loc.points_of_interest = points_of_interest
        if hidden_elements:
            loc.hidden_elements = hidden_elements
        if current_npcs:
            loc.current_npcs = current_npcs
        if make_current:
            campaign.current_location_id = loc.id
        await store.save(campaign)
        result = f"Set opening-scene detail for '{loc.name}' (id={loc.id})"
        return result + (" and marked it as the party's current location." if make_current else ".")

    return [set_opening_location_detail]


def make_tools(
    campaign_id: str,
    store: CampaignStore,
    lore_store: LoreStore | None = None,
    books_in_play: list[str] | None = None,
) -> list:
    """Full in-game world tool set — movement + authoring + travel."""
    return [
        *make_movement_tools(campaign_id, store),
        *make_authoring_tools(campaign_id, store, lore_store, books_in_play),
        *make_travel_tools(campaign_id, store),
    ]
