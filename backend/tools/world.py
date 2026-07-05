from langchain_core.tools import tool

from backend.models import AreaType, Location, LocationConnection, LocationScale, TravelTerrain
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import advance_clock, find_connection, find_container, find_location, opposite_direction


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


def make_authoring_tools(campaign_id: str, store: CampaignStore) -> list:

    @tool
    async def create_location(
        name: str,
        description: str | None = None,
        area_type: str = "outdoor",
        scale: str = "region",
        size: str | None = None,
        notes: str | None = None,
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
        place instead. Lighting, terrain features, and points of interest are
        added later once play begins, not during world-prep."""
        campaign = await store.load(campaign_id)
        existing = find_location(campaign, name)
        if existing:
            return f"Location '{existing.name}' already exists (id={existing.id}, scale={existing.scale.value}). Use connect_locations to link it."
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

    return [create_location, connect_locations]


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
        await store.save(campaign)
        npc_str = ", ".join(dest.current_npcs) if dest.current_npcs else "no one"
        return (
            f"The party travels {conn.distance_miles} mi to {dest.name} at {pace} pace "
            f"(~{hours:.0f} hours). {days_advanced} day(s) pass. "
            f"Now day {campaign.days_elapsed}, {campaign.time_of_day.value}. Present: {npc_str}."
        )

    return [get_travel_estimate, travel_to]


def make_tools(campaign_id: str, store: CampaignStore) -> list:
    """Full in-game world tool set — movement + authoring + travel."""
    return [
        *make_movement_tools(campaign_id, store),
        *make_authoring_tools(campaign_id, store),
        *make_travel_tools(campaign_id, store),
    ]
