from sqlalchemy import (
    Boolean, Column, Date, ForeignKey, Integer, MetaData, String, Table, Text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

metadata = MetaData()


# ── Root ──────────────────────────────────────────────────────────────────────

# Flat columns on campaigns are the only ones read without loading `data` —
# used by list_all() to populate the campaign selector without parsing blobs.
campaigns = Table(
    "campaigns", metadata,
    Column("id",         UUID(as_uuid=False), primary_key=True),
    Column("name",       Text,  nullable=False),
    Column("created_at", Date,  nullable=False),
    # All other Campaign fields live here: setting, books_in_play,
    # system_prompt_variant, current_location_id, time tracking,
    # party_treasury, notes, etc.
    Column("data",       JSONB, nullable=False),
)


# ── Entity tables ─────────────────────────────────────────────────────────────
# Every entity table follows the same pattern:
#   - id + campaign_id  (structural)
#   - key flat columns  (queryable without touching data)
#   - data JSONB        (full Pydantic model — source of truth for load)

def _entity(name: str, *extra: Column) -> Table:
    return Table(
        name, metadata,
        Column("id",          UUID(as_uuid=False), primary_key=True),
        Column("campaign_id", UUID(as_uuid=False),
               ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        *extra,
        Column("data", JSONB, nullable=False),
    )


characters = _entity(
    "characters",
    Column("name",                Text,    nullable=False),
    Column("char_class",          Text,    nullable=False),
    Column("level",               Integer, nullable=False),
    Column("current_hp",          Integer, nullable=False),
    Column("max_hp",              Integer, nullable=False),
    Column("ac",                  Integer, nullable=False),
    Column("is_player_controlled", Boolean, nullable=False),
)

monsters = _entity(
    "monsters",
    Column("name",       Text,    nullable=False),
    Column("cr",         Text,    nullable=False),
    Column("current_hp", Integer, nullable=False),
    Column("max_hp",     Integer, nullable=False),
    Column("ac",         Integer, nullable=False),
)

npcs = _entity(
    "npcs",
    Column("name",          Text,    nullable=False),
    Column("is_alive",      Boolean, nullable=False),
    Column("has_met_party", Boolean, nullable=False),
    Column("attitude",      String,  nullable=False),
    Column("faction_id",    UUID(as_uuid=False), nullable=True),
)

factions = _entity(
    "factions",
    Column("name",              Text,    nullable=False),
    Column("party_reputation",  Integer, nullable=False),
)

quests = _entity(
    "quests",
    Column("name",       Text, nullable=False),
    Column("quest_type", Text, nullable=False),
    Column("status",     Text, nullable=False),
)

locations = _entity(
    "locations",
    Column("name",      Text, nullable=False),
    Column("area_type", Text, nullable=False),
    Column("lighting",  Text, nullable=False),
)

containers = _entity(
    "containers",
    Column("name",      Text,    nullable=False),
    Column("is_locked", Boolean, nullable=False),
    Column("is_open",   Boolean, nullable=False),
)

traps = _entity(
    "traps",
    Column("name",         Text,    nullable=False),
    Column("is_detected",  Boolean, nullable=False),
    Column("is_triggered", Boolean, nullable=False),
)

handouts = _entity(
    "handouts",
    Column("title",                Text,    nullable=False),
    Column("handout_type",         Text,    nullable=False),
    Column("is_revealed_to_party", Boolean, nullable=False),
)

sessions = _entity(
    "sessions",
    Column("session_number", Integer, nullable=False),
    Column("real_date",      Date,    nullable=True),
    Column("xp_awarded",     Integer, nullable=False),
)

encounters = _entity(
    "encounters",
    Column("is_active",  Boolean, nullable=False),
    Column("round",      Integer, nullable=False),
    Column("difficulty", Text,    nullable=False),
)

# Append-only roll log — no data JSONB needed, all fields are flat.
rolls = Table(
    "rolls", metadata,
    Column("id",          UUID(as_uuid=False), primary_key=True),
    Column("campaign_id", UUID(as_uuid=False),
           ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
    Column("notation",   Text,                   nullable=False),
    Column("result",     Integer,                nullable=False),
    Column("breakdown",  Text,                   nullable=False),
    Column("rolled_at",  TIMESTAMP(timezone=True), nullable=False),
)
