"""Coverage for backend/session_export.py's render_session_export_markdown —
the "Session summary export (markdown or PDF)" Planned Future Feature, markdown
half. Pure function, no DB dependency (the live route at
GET /campaigns/{id}/sessions/export/markdown is a thin load+render wrapper
around this, verified manually against a real campaign)."""
from backend.models import Campaign, Item, Quest, Session
from backend.session_export import render_session_export_markdown


def test_empty_campaign_has_no_sessions_placeholder():
    campaign = Campaign(id="c1", name="Empty Campaign")
    out = render_session_export_markdown(campaign)
    assert "# Empty Campaign — Session Log" in out
    assert "No sessions recorded yet" in out


def test_sessions_are_ordered_by_session_number_not_list_order():
    campaign = Campaign(
        id="c1", name="Test Campaign",
        sessions=[
            Session(session_number=2, summary="Second session."),
            Session(session_number=1, summary="First session."),
        ],
    )
    out = render_session_export_markdown(campaign)
    assert out.index("Session 1") < out.index("Session 2")
    assert out.index("First session.") < out.index("Second session.")


def test_full_session_fields_all_render():
    campaign = Campaign(
        id="c1", name="Yawning Portal",
        quests=[Quest(id="q1", name="Find the Lost Amulet")],
        sessions=[
            Session(
                session_number=1, summary="The party met in a tavern.",
                key_events=["Met Volo", "Found a map"],
                adventure_progress="Chapter 1: The Beginning", xp_awarded=300,
                loot_gained=[Item(name="Potion of Healing", quantity=2)],
                quests_started=["q1"], notes="Great first session.",
            ),
        ],
    )
    out = render_session_export_markdown(campaign)
    assert "The party met in a tavern." in out
    assert "- Met Volo" in out
    assert "- Found a map" in out
    assert "Chapter 1: The Beginning" in out
    assert "**XP awarded:** 300" in out
    assert "Potion of Healing x2" in out
    # Quest ID resolved to its real name, not the raw ID
    assert "Find the Lost Amulet" in out
    assert "q1" not in out.replace("Find the Lost Amulet", "")
    assert "Great first session." in out


def test_a_quest_id_with_no_matching_quest_falls_back_to_the_raw_id():
    campaign = Campaign(
        id="c1", name="Test Campaign",
        sessions=[Session(session_number=1, quests_completed=["missing-quest-id"])],
    )
    out = render_session_export_markdown(campaign)
    assert "missing-quest-id" in out
