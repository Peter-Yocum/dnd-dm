from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import Table, delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.models import (
    Campaign, Character, Container, Encounter, Faction, Handout,
    Location, Monster, NPC, Quest, Session, Trap,
)
from backend.stores import tables as t


# ── Public value types returned by the store ──────────────────────────────────

class CampaignSummary(BaseModel):
    id: str
    name: str
    created_at: str   # ISO date string — avoids date/datetime ambiguity across callers


class RollEntry(BaseModel):
    id: str
    notation: str
    result: int
    breakdown: str
    rolled_at: datetime


# ── Row builders ──────────────────────────────────────────────────────────────
# Each function maps a Pydantic model to the dict expected by its table.
# Flat columns duplicate key fields so SQL queries can filter without
# deserialising the data JSONB blob.

def _campaign_data(campaign: Campaign) -> dict:
    """Extract the non-entity, non-identity fields that live in campaigns.data."""
    return campaign.model_dump(
        mode="json",
        exclude={
            "id", "name", "created_at",
            "party", "monsters", "npcs", "factions", "quests",
            "locations", "containers", "traps", "handouts", "sessions",
            "active_encounter",
        },
    )


def _char_row(c: Character, campaign_id: str) -> dict:
    return {
        "id": c.id, "campaign_id": campaign_id,
        "name": c.name, "char_class": c.char_class, "level": c.level,
        "current_hp": c.current_hp, "max_hp": c.max_hp, "ac": c.ac,
        "is_player_controlled": c.is_player_controlled,
        "data": c.model_dump(mode="json"),
    }


def _monster_row(m: Monster, campaign_id: str) -> dict:
    return {
        "id": m.id, "campaign_id": campaign_id,
        "name": m.name, "cr": m.cr,
        "current_hp": m.current_hp, "max_hp": m.max_hp, "ac": m.ac,
        "data": m.model_dump(mode="json"),
    }


def _npc_row(n: NPC, campaign_id: str) -> dict:
    return {
        "id": n.id, "campaign_id": campaign_id,
        "name": n.name, "is_alive": n.is_alive,
        "has_met_party": n.has_met_party, "attitude": n.attitude.value,
        "faction_id": n.faction_id,
        "data": n.model_dump(mode="json"),
    }


def _faction_row(f: Faction, campaign_id: str) -> dict:
    return {
        "id": f.id, "campaign_id": campaign_id,
        "name": f.name, "party_reputation": f.party_reputation,
        "data": f.model_dump(mode="json"),
    }


def _quest_row(q: Quest, campaign_id: str) -> dict:
    return {
        "id": q.id, "campaign_id": campaign_id,
        "name": q.name, "quest_type": q.quest_type.value, "status": q.status.value,
        "data": q.model_dump(mode="json"),
    }


def _location_row(loc: Location, campaign_id: str) -> dict:
    return {
        "id": loc.id, "campaign_id": campaign_id,
        "name": loc.name, "area_type": loc.area_type.value, "lighting": loc.lighting.value,
        "data": loc.model_dump(mode="json"),
    }


def _container_row(c: Container, campaign_id: str) -> dict:
    return {
        "id": c.id, "campaign_id": campaign_id,
        "name": c.name, "is_locked": c.is_locked, "is_open": c.is_open,
        "data": c.model_dump(mode="json"),
    }


def _trap_row(tr: Trap, campaign_id: str) -> dict:
    return {
        "id": tr.id, "campaign_id": campaign_id,
        "name": tr.name, "is_detected": tr.is_detected, "is_triggered": tr.is_triggered,
        "data": tr.model_dump(mode="json"),
    }


def _handout_row(h: Handout, campaign_id: str) -> dict:
    return {
        "id": h.id, "campaign_id": campaign_id,
        "title": h.title, "handout_type": h.handout_type.value,
        "is_revealed_to_party": h.is_revealed_to_party,
        "data": h.model_dump(mode="json"),
    }


def _session_row(s: Session, campaign_id: str) -> dict:
    return {
        "id": s.id, "campaign_id": campaign_id,
        "session_number": s.session_number,
        "real_date": s.real_date,
        "xp_awarded": s.xp_awarded,
        "data": s.model_dump(mode="json"),
    }


def _encounter_row(enc: Encounter, campaign_id: str) -> dict:
    return {
        "id": enc.id, "campaign_id": campaign_id,
        "is_active": enc.is_active, "round": enc.round,
        "difficulty": enc.difficulty.value,
        "data": enc.model_dump(mode="json"),
    }


# ── Store ─────────────────────────────────────────────────────────────────────

class CampaignStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _fetch_entities(self, conn, table, campaign_id: str) -> list:
        result = await conn.execute(
            select(table).where(table.c.campaign_id == campaign_id)
        )
        return result.mappings().all()

    async def _write_entities(self, conn, campaign: Campaign) -> None:
        """Delete and reinsert all entity rows for this campaign inside an open txn."""
        cid = campaign.id
        entity_tables = [
            t.characters, t.monsters, t.npcs, t.factions, t.quests,
            t.locations, t.containers, t.traps, t.handouts, t.sessions,
            t.encounters,
        ]
        for table in entity_tables:
            await conn.execute(delete(table).where(table.c.campaign_id == cid))

        async def insert_many(table: Table, rows: list[dict]) -> None:
            if rows:
                await conn.execute(insert(table), rows)

        await insert_many(t.characters,  [_char_row(c, cid)      for c in campaign.party])
        await insert_many(t.monsters,    [_monster_row(m, cid)    for m in campaign.monsters])
        await insert_many(t.npcs,        [_npc_row(n, cid)        for n in campaign.npcs])
        await insert_many(t.factions,    [_faction_row(f, cid)    for f in campaign.factions])
        await insert_many(t.quests,      [_quest_row(q, cid)      for q in campaign.quests])
        await insert_many(t.locations,   [_location_row(l, cid)   for l in campaign.locations])
        await insert_many(t.containers,  [_container_row(c, cid)  for c in campaign.containers])
        await insert_many(t.traps,       [_trap_row(tr, cid)      for tr in campaign.traps])
        await insert_many(t.handouts,    [_handout_row(h, cid)    for h in campaign.handouts])
        await insert_many(t.sessions,    [_session_row(s, cid)    for s in campaign.sessions])

        if campaign.active_encounter:
            await conn.execute(
                insert(t.encounters),
                [_encounter_row(campaign.active_encounter, cid)],
            )

    # ── Campaign CRUD ─────────────────────────────────────────────────────────

    async def create(self, campaign: Campaign) -> Campaign:
        async with self._engine.begin() as conn:
            await conn.execute(insert(t.campaigns).values(
                id=campaign.id,
                name=campaign.name,
                created_at=campaign.created_at,
                data=_campaign_data(campaign),
            ))
            await self._write_entities(conn, campaign)
        return campaign

    async def load(self, campaign_id: str) -> Campaign | None:
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                select(t.campaigns).where(t.campaigns.c.id == campaign_id)
            )).mappings().first()
            if row is None:
                return None

            char_rows     = await self._fetch_entities(conn, t.characters,  campaign_id)
            monster_rows  = await self._fetch_entities(conn, t.monsters,    campaign_id)
            npc_rows      = await self._fetch_entities(conn, t.npcs,        campaign_id)
            faction_rows  = await self._fetch_entities(conn, t.factions,    campaign_id)
            quest_rows    = await self._fetch_entities(conn, t.quests,      campaign_id)
            location_rows = await self._fetch_entities(conn, t.locations,   campaign_id)
            container_rows= await self._fetch_entities(conn, t.containers,  campaign_id)
            trap_rows     = await self._fetch_entities(conn, t.traps,       campaign_id)
            handout_rows  = await self._fetch_entities(conn, t.handouts,    campaign_id)
            session_rows  = await self._fetch_entities(conn, t.sessions,    campaign_id)
            encounter_rows= await self._fetch_entities(conn, t.encounters,  campaign_id)

        active = next((r for r in encounter_rows if r["is_active"]), None)

        return Campaign(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            **row["data"],
            party=     [Character.model_validate(r["data"]) for r in char_rows],
            monsters=  [Monster.model_validate(r["data"])   for r in monster_rows],
            npcs=      [NPC.model_validate(r["data"])       for r in npc_rows],
            factions=  [Faction.model_validate(r["data"])   for r in faction_rows],
            quests=    [Quest.model_validate(r["data"])     for r in quest_rows],
            locations= [Location.model_validate(r["data"])  for r in location_rows],
            containers=[Container.model_validate(r["data"]) for r in container_rows],
            traps=     [Trap.model_validate(r["data"])      for r in trap_rows],
            handouts=  [Handout.model_validate(r["data"])   for r in handout_rows],
            sessions=  [Session.model_validate(r["data"])   for r in session_rows],
            active_encounter=Encounter.model_validate(active["data"]) if active else None,
        )

    async def save(self, campaign: Campaign) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                update(t.campaigns)
                .where(t.campaigns.c.id == campaign.id)
                .values(name=campaign.name, data=_campaign_data(campaign))
            )
            await self._write_entities(conn, campaign)

    async def list_all(self) -> list[CampaignSummary]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(t.campaigns.c.id, t.campaigns.c.name, t.campaigns.c.created_at)
                .order_by(t.campaigns.c.created_at.desc())
            )).mappings().all()
        return [
            CampaignSummary(id=r["id"], name=r["name"], created_at=str(r["created_at"]))
            for r in rows
        ]

    async def delete(self, campaign_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                delete(t.campaigns).where(t.campaigns.c.id == campaign_id)
            )  # ON DELETE CASCADE removes all entity rows automatically

    # ── Roll log ──────────────────────────────────────────────────────────────

    async def log_roll(
        self, campaign_id: str, notation: str, result: int, breakdown: str
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(insert(t.rolls).values(
                id=uuid4().hex,
                campaign_id=campaign_id,
                notation=notation,
                result=result,
                breakdown=breakdown,
                rolled_at=datetime.now(tz=timezone.utc),
            ))

    async def get_rolls(self, campaign_id: str, limit: int = 50) -> list[RollEntry]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(t.rolls)
                .where(t.rolls.c.campaign_id == campaign_id)
                .order_by(t.rolls.c.rolled_at.desc())
                .limit(limit)
            )).mappings().all()
        return [
            RollEntry(
                id=r["id"], notation=r["notation"], result=r["result"],
                breakdown=r["breakdown"], rolled_at=r["rolled_at"],
            )
            for r in rows
        ]
