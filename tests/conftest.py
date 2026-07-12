"""Shared fixtures for the tool-layer BDD suite.

Runs against the same dev Postgres the app itself uses (backend.config.settings) —
there's no separate test database in this project. Every fixture that creates a
Campaign row cleans it up itself, so this is safe to run against a real dev
instance without leaving scratch data behind.
"""
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.models import (
    AbilityScores, Campaign, Character, CombatantType, Encounter, InitiativeEntry, Monster,
)
from backend.stores.campaign_store import CampaignStore


@pytest_asyncio.fixture
async def store():
    engine = create_async_engine(settings.database_url)
    yield CampaignStore(engine)
    await engine.dispose()


@pytest_asyncio.fixture
async def campaign(store):
    """An empty, saved Campaign — tests populate party/monsters/encounter
    themselves and call store.save(campaign) again after mutating it."""
    c = Campaign(id=str(uuid.uuid4()), name="bdd-test-campaign")
    await store.create(c)
    yield c
    await store.delete(c.id)


@pytest.fixture
def force_hit(monkeypatch):
    """Pins every d20 roll in resolution.py to a natural 20 — attack-economy
    tests care about whether a call is ALLOWED (budget/turn checks), not
    about genuine 5e hit/miss odds, and a real 1-in-20 fumble would otherwise
    make them flaky (a nat 1 always misses regardless of AC, per
    resolution.py's _roll_to_hit)."""
    monkeypatch.setattr("backend.tools.resolution.random.randint", lambda a, b: 20)


def make_character(name: str, **overrides) -> Character:
    defaults = dict(
        name=name, char_class="Fighter", level=1, max_hp=12, current_hp=12,
        ability_scores=AbilityScores(), ac=15,
    )
    defaults.update(overrides)
    return Character(**defaults)


def make_monster(name: str, **overrides) -> Monster:
    defaults = dict(name=name, max_hp=7, current_hp=7, ac=13)
    defaults.update(overrides)
    return Monster(**defaults)


def start_combat(campaign: Campaign, combatants: list[tuple[str, str, int]]) -> Encounter:
    """combatants: list of (name, combatant_type, initiative), first entry
    becomes the current turn. combatant_type is "character" | "monster"."""
    order = [
        InitiativeEntry(
            name=name, combatant_type=CombatantType(ctype), initiative=init,
            is_current_turn=(i == 0),
        )
        for i, (name, ctype, init) in enumerate(combatants)
    ]
    enc = Encounter(is_active=True, round=1, initiative_order=order)
    campaign.active_encounter = enc
    return enc
