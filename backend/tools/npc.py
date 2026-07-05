from langchain_core.tools import tool

from backend.models import Attitude, Campaign, NPC
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import find_location, find_npc


def build_traveling_npcs_context(campaign: Campaign) -> str | None:
    """Always-on reminder of NPCs currently traveling with the party — the only
    NPCs not reliably re-surfaced each turn via Location.current_npcs, which is
    tied to a fixed place and isn't auto-injected the way encounter state is
    (see build_encounter_context in combat.py). Keeps a recruited ally from
    silently dropping out of context once conversation history trims past
    their introduction."""
    npcs = [n for n in campaign.npcs if n.traveling_with_party]
    if not npcs:
        return None
    lines = ["Traveling with the party:"]
    for n in npcs:
        lines.append(f"  {n.name} ({n.race} {n.occupation}) — {n.attitude.value}")
    return "\n".join(lines)


def make_tools(campaign_id: str, store: CampaignStore) -> list:

    @tool
    async def get_npc(npc_name: str) -> str:
        """Get an NPC's attitude, motivations, knowledge, and secrets. Call before
        roleplaying any named NPC to ground their behaviour in what the campaign
        record says about them."""
        campaign = await store.load(campaign_id)
        npc = find_npc(campaign, npc_name)
        if not npc:
            return f"No NPC named '{npc_name}' in this campaign."
        lines = [
            f"=== {npc.name} ({npc.race} {npc.occupation}) ===",
            f"Location: {npc.location or 'unknown'}",
            f"Alive: {npc.is_alive}  Met party: {npc.has_met_party}  Attitude: {npc.attitude.value}",
        ]
        if npc.faction_id:
            faction = next((f for f in campaign.factions if f.id == npc.faction_id), None)
            if faction:
                lines.append(f"Faction: {faction.name}")
        if npc.personality_traits:
            lines.append(f"Traits: {'; '.join(npc.personality_traits)}")
        if npc.ideals:
            lines.append(f"Ideals: {'; '.join(npc.ideals)}")
        if npc.bonds:
            lines.append(f"Bonds: {'; '.join(npc.bonds)}")
        if npc.flaws:
            lines.append(f"Flaws: {'; '.join(npc.flaws)}")
        if npc.motivations:
            lines.append("Motivations:")
            lines += [f"  - {m}" for m in npc.motivations]
        if npc.knowledge:
            lines.append("Will share if asked:")
            lines += [f"  [{i}] {k}" for i, k in enumerate(npc.knowledge)]
        if npc.secrets:
            lines.append("Secrets (won't volunteer):")
            lines += [f"  - {s}" for s in npc.secrets]
        if npc.relationships:
            lines.append("Relationships: " + ", ".join(
                f"{r.npc_name} ({r.description})" for r in npc.relationships
            ))
        if npc.is_merchant:
            lines.append(f"Merchant (price modifier: {npc.price_modifier}x)")
        if npc.inventory:
            lines.append("Carries: " + ", ".join(i.name for i in npc.inventory))
        if npc.notes:
            lines.append(f"Notes: {npc.notes}")
        return "\n".join(lines)

    @tool
    async def update_npc_attitude(npc_name: str, attitude: str) -> str:
        """Change an NPC's attitude toward the party after a significant interaction.
        Valid attitudes: friendly, helpful, indifferent, cautious, unfriendly,
        suspicious, hostile, fearful."""
        campaign = await store.load(campaign_id)
        npc = find_npc(campaign, npc_name)
        if not npc:
            return f"No NPC named '{npc_name}' in this campaign."
        try:
            new_attitude = Attitude(attitude.lower())
        except ValueError:
            return f"'{attitude}' is not a valid attitude. Choose from: {', '.join(a.value for a in Attitude)}."
        old = npc.attitude.value
        npc.attitude = new_attitude
        npc.has_met_party = True
        await store.save(campaign)
        return f"{npc.name}'s attitude changed: {old} → {new_attitude.value}."

    @tool
    async def reveal_npc_knowledge(npc_name: str, knowledge_index: int) -> str:
        """Mark a piece of NPC knowledge as shared with the party. Use the index
        shown by get_npc. Call after the NPC actually tells the party something."""
        campaign = await store.load(campaign_id)
        npc = find_npc(campaign, npc_name)
        if not npc:
            return f"No NPC named '{npc_name}' in this campaign."
        if not (0 <= knowledge_index < len(npc.knowledge)):
            return f"{npc.name} has no knowledge entry at index {knowledge_index}."
        revealed = npc.knowledge[knowledge_index]
        # Move to notes so it's flagged as shared but still accessible
        npc.notes = (npc.notes + f"\n[REVEALED TO PARTY] {revealed}").strip()
        npc.knowledge.pop(knowledge_index)
        await store.save(campaign)
        return f"{npc.name} told the party: '{revealed}'."

    @tool
    async def create_npc(
        name: str,
        race: str = "",
        occupation: str = "",
        attitude: str = "indifferent",
        location: str = "",
        notes: str = "",
    ) -> str:
        """Create a new NPC and add them to the campaign. Use when the party
        encounters someone not yet in the campaign record — a shopkeeper, a
        random guard, a bystander the players decided to befriend."""
        campaign = await store.load(campaign_id)
        if find_npc(campaign, name):
            return f"An NPC named '{name}' already exists."
        try:
            att = Attitude(attitude.lower())
        except ValueError:
            att = Attitude.INDIFFERENT
        npc = NPC(
            name=name,
            race=race,
            occupation=occupation,
            attitude=att,
            location=location,
            notes=notes,
        )
        campaign.npcs.append(npc)
        await store.save(campaign)
        return f"Created NPC '{name}' ({race} {occupation}, {att.value}) in {location or 'unknown location'}."

    @tool
    async def set_npc_traveling_with_party(npc_name: str, traveling: bool = True) -> str:
        """Mark an NPC as currently traveling with the party (or no longer). Call
        once an NPC is recruited/joins the group for the road, so they keep showing
        up in context regardless of which Location they're nominally tied to. Call
        with traveling=False once they leave the party/stay behind."""
        campaign = await store.load(campaign_id)
        npc = find_npc(campaign, npc_name)
        if not npc:
            return f"No NPC named '{npc_name}' in this campaign."
        npc.traveling_with_party = traveling
        await store.save(campaign)
        return f"{npc.name} is {'now' if traveling else 'no longer'} traveling with the party."

    @tool
    async def place_npc_at_location(npc_name: str, location_name: str) -> str:
        """Place an NPC at a location independent of where the party currently is —
        for an NPC who's leaving the party now but will turn up somewhere specific
        later (e.g. "if you freed me, I'll be waiting at the tavern in Greensdale").
        Adds them to that location's current_npcs so a normal get_current_location
        call surfaces them the moment the party arrives, with no reliance on
        remembering this conversation or re-deriving it from search later. The
        location must already exist — call create_location first if it doesn't."""
        campaign = await store.load(campaign_id)
        npc = find_npc(campaign, npc_name)
        loc = find_location(campaign, location_name)
        if not npc:
            return f"No NPC named '{npc_name}' in this campaign."
        if not loc:
            return f"No location named '{location_name}' found — create it first if it doesn't exist yet."
        if npc.name not in loc.current_npcs:
            loc.current_npcs.append(npc.name)
        npc.location = loc.name
        await store.save(campaign)
        return f"{npc.name} is now placed at {loc.name} — will surface via get_current_location once the party arrives."

    return [
        get_npc, update_npc_attitude, reveal_npc_knowledge, create_npc,
        set_npc_traveling_with_party, place_npc_at_location,
    ]
