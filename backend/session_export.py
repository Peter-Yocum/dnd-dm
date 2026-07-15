"""Session summary export — renders a campaign's recorded Sessions
(Campaign.sessions, populated by summarize_session() at session end) into a
single downloadable markdown document. No PDF output yet (see design.md's
Planned Future Features note) — markdown alone satisfies the "export a
readable session log" need without a new rendering dependency; a PDF is a
straightforward follow-up (pipe this same markdown through a converter)
if it's ever actually wanted.

Pure string-building, no I/O — kept separate from main.py so it's testable
without a DB/HTTP round-trip.
"""

from backend.models import Campaign, Quest, Session


def _quest_name(campaign: Campaign, quest_id: str) -> str:
    quest = next((q for q in campaign.quests if q.id == quest_id), None)
    return quest.name if quest else quest_id


def render_session_export_markdown(campaign: Campaign) -> str:
    """One markdown document covering every recorded Session, oldest first —
    the same data `sessions.html` renders per-session, just flattened into
    a single exportable file."""
    sessions = sorted(campaign.sessions, key=lambda s: s.session_number)

    lines = [f"# {campaign.name} — Session Log", ""]
    if campaign.setting:
        lines.append(f"*{campaign.setting}*")
        lines.append("")
    if not sessions:
        lines.append("_No sessions recorded yet._")
        return "\n".join(lines) + "\n"

    for session in sessions:
        lines.append(f"## Session {session.session_number}" + (f" — {session.real_date}" if session.real_date else ""))
        lines.append("")

        if session.summary:
            lines.append(session.summary)
            lines.append("")

        if session.key_events:
            lines.append("**Key events:**")
            lines.extend(f"- {event}" for event in session.key_events)
            lines.append("")

        if session.adventure_progress:
            lines.append(f"**Adventure progress:** {session.adventure_progress}")
            lines.append("")

        if session.xp_awarded:
            lines.append(f"**XP awarded:** {session.xp_awarded}")
            lines.append("")

        if session.loot_gained:
            items = ", ".join(
                f"{item.name}" + (f" x{item.quantity}" if item.quantity > 1 else "")
                for item in session.loot_gained
            )
            lines.append(f"**Loot gained:** {items}")
            lines.append("")

        if session.quests_started:
            names = ", ".join(_quest_name(campaign, q) for q in session.quests_started)
            lines.append(f"**Quests started:** {names}")
            lines.append("")

        if session.quests_completed:
            names = ", ".join(_quest_name(campaign, q) for q in session.quests_completed)
            lines.append(f"**Quests completed:** {names}")
            lines.append("")

        if session.notes:
            lines.append(f"**Notes:** {session.notes}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
