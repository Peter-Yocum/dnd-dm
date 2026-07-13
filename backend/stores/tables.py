from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, Column, Date, ForeignKey, Integer, MetaData, String, Table, Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, TSVECTOR, UUID

metadata = MetaData()

# mlx-community/Qwen3-Embedding-0.6B-8bit's real output dimension — confirmed
# live against a running vllm-metal --convert embed instance (2026-07-13),
# not assumed from documentation. Was 768 (nomic-embed-text, via Ollama)
# before the vLLM-metal embeddings migration (vllm-migration-plan.md §7.7) —
# a breaking change requiring every existing embedding to be regenerated,
# not just a config bump (see that section's migration notes).
EMBEDDING_DIM = 1024


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

# ── Lore Registry (canon — book-scoped, NOT campaign-scoped) ──────────────────
# Populated offline by scripts/extract_entities.py --write-postgres. Read by
# lookup_entity/search_lore (backend/tools/lore.py) and world_prep.py. Never
# mutated by live play — a campaign's own NPC/Location/Item rows point back
# at these via lore_entity_id (provenance), but this table itself only
# changes when a book is (re-)extracted.

lore_entities = Table(
    "lore_entities", metadata,
    Column("id",                UUID(as_uuid=False), primary_key=True),
    Column("book_slug",         Text, nullable=False),
    Column("source_type",       Text, nullable=False, server_default="adventure"),  # "core" | "adventure"
    Column("entity_type",       Text, nullable=False),  # "npc" | "location" | "item" | "monster"
    Column("canonical_name",    Text, nullable=False),
    Column("rolled_up_profile", JSONB, nullable=False),
    Column("source_chunk_ids",  ARRAY(Text), nullable=False, server_default="{}"),
    Column("spoiler_tier",      Text, nullable=False, server_default="public"),
    Column("created_at",        TIMESTAMP(timezone=True), nullable=False),
    UniqueConstraint("book_slug", "entity_type", "canonical_name", name="uq_lore_entities_book_type_name"),
)

lore_entity_aliases = Table(
    "lore_entity_aliases", metadata,
    Column("id",              UUID(as_uuid=False), primary_key=True),
    Column("lore_entity_id",  UUID(as_uuid=False),
           ForeignKey("lore_entities.id", ondelete="CASCADE"), nullable=False),
    Column("alias",           Text, nullable=False),
    UniqueConstraint("lore_entity_id", "alias", name="uq_lore_entity_aliases_entity_alias"),
)

# ── Incremental relation graph (campaign-scoped — see Stage 1.5) ─────────────
# Edges between entities (NPC<->location, NPC<->faction, item<->location, ...).
# UniqueConstraint on (campaign_id, source_id, target_id, relation) IS the
# set-merging mechanism: re-adding the same fact is a harmless upsert no-op.

entity_relations = Table(
    "entity_relations", metadata,
    Column("id",               UUID(as_uuid=False), primary_key=True),
    Column("campaign_id",      UUID(as_uuid=False),
           ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
    Column("source_type",      Text, nullable=False),  # npc | location | item | faction | quest
    Column("source_id",        Text, nullable=False),
    Column("source_name",      Text, nullable=False),
    Column("target_type",      Text, nullable=False),
    Column("target_id",        Text, nullable=False),
    Column("target_name",      Text, nullable=False),
    Column("relation",         Text, nullable=False),  # "member of", "located in", "allied with", ...
    Column("description",      Text, nullable=False, server_default=""),
    Column("source_chunk_ids", ARRAY(Text), nullable=False, server_default="{}"),
    Column("created_at",       TIMESTAMP(timezone=True), nullable=False),
    UniqueConstraint("campaign_id", "source_id", "target_id", "relation", name="uq_entity_relations_edge"),
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


# ── Rules corpus (core rulebooks + adventures) ─────────────────────────────────
# Replaces ChromaDB's "rules" collection (2026-07-12 migration — see design.md's
# Tech Stack table and that date's migration plan for why: Chroma's own local
# vector index needed ~2.6GB resident just to query a 441k-row collection,
# independent of any keyword-index work; pgvector + native full-text search
# consolidates both onto the one DB this app already uses for everything else).
# Book-agnostic/campaign-agnostic — shared across every campaign, populated
# offline by scripts/build_index.py. `embedding` is NULL for "parent" rows
# (never embedded/searched directly, only fetched by id to expand a matched
# child chunk back to its full section — see backend/stores/rules_store.py).
rule_chunks = Table(
    "rule_chunks", metadata,
    Column("chunk_id",        Text, primary_key=True),  # deterministic content hash, build_index.py's _chunk_id
    Column("book",            Text, nullable=False),
    Column("section",         Text, nullable=False),
    Column("source_type",     Text, nullable=False),      # "core" | "adventure"
    Column("adventure",       Text, nullable=False, server_default=""),
    Column("granularity",     Text, nullable=False),       # "child" | "parent"
    Column("parent_chunk_id", Text, nullable=False, server_default=""),
    Column("sequence_number", Integer, nullable=False, server_default="0"),
    Column("content",         Text, nullable=False),
    Column("embedding",       Vector(EMBEDDING_DIM), nullable=True),
    Column("content_tsv",     TSVECTOR, nullable=True),  # GENERATED ALWAYS AS column, see migration 0004
    # True once this row's embedding was computed from the contextualized
    # (blurb-prefixed) text, not just the raw content — lets a later
    # --recontextualize pass resume mid-corpus instead of redoing everything
    # --force would. See migration 0005.
    Column("contextualized",  Boolean, nullable=False, server_default="false"),
)

# ── Session chronicles (per-campaign play history) ─────────────────────────────
# Replaces ChromaDB's "session_chronicles" collection. Fully regenerable from
# Campaign.sessions (JSONB in `campaigns.data`) — see backend/stores/history_store.py.
session_chronicle_chunks = Table(
    "session_chronicle_chunks", metadata,
    Column("chunk_id",       Text, primary_key=True),  # f"{session_id}::{event_type}::{i}"
    Column("campaign_id",    UUID(as_uuid=False),
           ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
    Column("session_id",     Text, nullable=False),
    Column("session_number", Integer, nullable=False),
    Column("event_index",    Integer, nullable=False),
    Column("event_type",     Text, nullable=False),      # "summary" | "key_event"
    Column("content",        Text, nullable=False),
    Column("embedding",      Vector(EMBEDDING_DIM), nullable=True),
    Column("content_tsv",    TSVECTOR, nullable=True),
)
