from datetime import date

from langchain_core.tools import tool

from backend.models import QuestStatus, Session
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import read_adventure_meta


def make_tools(campaign_id: str, store: CampaignStore) -> list:

    @tool
    async def get_campaign_summary() -> str:
        """Return a high-level overview of the campaign — setting, party,
        active quests, current location, in-game date, and session count.
        Useful at session start or when the DM needs a quick orientation."""
        campaign = await store.load(campaign_id)
        lines = [
            f"Campaign: {campaign.name}",
            f"Setting: {campaign.setting or 'not set'}",
            f"Sessions run: {campaign.session_count}",
            f"In-game date: {campaign.in_game_date or 'not tracked'}  Time: {campaign.time_of_day.value}",
            f"Days elapsed: {campaign.days_elapsed}  Weather: {campaign.current_weather or 'clear'}",
        ]
        if campaign.current_location_id:
            loc = next((l for l in campaign.locations if l.id == campaign.current_location_id), None)
            lines.append(f"Current location: {loc.name if loc else '(unknown)'}")
        if campaign.party:
            lines.append(f"Party ({len(campaign.party)} member(s)):")
            for c in campaign.party:
                role = "PC" if c.is_player_controlled else "companion"
                lines.append(f"  {c.name} — {c.race} {c.char_class} {c.level} [{role}]  {c.current_hp}/{c.max_hp} HP")
        else:
            lines.append("Party: empty")
        for slug in campaign.books_in_play:
            rec = read_adventure_meta(slug).get("recommended_players")
            if rec:
                lines.append(f"Recommended party size for {slug}: {rec} "
                              f"(currently {len(campaign.party)} — use generate_companion_character "
                              f"to add a DM-controlled companion if short, choosing a build that "
                              f"complements the existing party rather than duplicating it)")
        active_quests = [q for q in campaign.quests if q.status == QuestStatus.ACTIVE]
        if active_quests:
            lines.append("Active quests: " + ", ".join(q.name for q in active_quests))
        if campaign.active_encounter and campaign.active_encounter.is_active:
            enc = campaign.active_encounter
            current = next((e for e in enc.initiative_order if e.is_current_turn), None)
            lines.append(f"IN COMBAT — round {enc.round}, {current.name}'s turn" if current else f"IN COMBAT — round {enc.round}")
        if campaign.notes:
            lines.append(f"Campaign notes: {campaign.notes}")
        return "\n".join(lines)

    @tool
    async def add_session_note(note: str) -> str:
        """Append an important event or note to the current session log.
        Creates a new session record if none is active. Use for significant
        moments: a major reveal, a PC decision, loot found, an alliance formed."""
        campaign = await store.load(campaign_id)

        # Find or create the current session
        if campaign.sessions:
            session = campaign.sessions[-1]
        else:
            campaign.session_count += 1
            session = Session(
                session_number=campaign.session_count,
                real_date=date.today(),
            )
            campaign.sessions.append(session)

        session.key_events.append(note)
        await store.save(campaign)
        return f"[Session {session.session_number}] Note recorded: {note}"

    return [get_campaign_summary, add_session_note]
