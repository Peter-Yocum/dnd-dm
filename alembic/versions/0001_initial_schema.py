"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-29
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "campaigns",
        sa.Column("id",         UUID(as_uuid=False), primary_key=True),
        sa.Column("name",       sa.Text(),  nullable=False),
        sa.Column("created_at", sa.Date(),  nullable=False),
        sa.Column("data",       JSONB(),    nullable=False, server_default="{}"),
    )

    # ── Entity tables (all reference campaigns.id with CASCADE delete) ────────

    def entity(name: str, *extra):
        return op.create_table(
            name,
            sa.Column("id",          UUID(as_uuid=False), primary_key=True),
            sa.Column("campaign_id", UUID(as_uuid=False),
                      sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
            *extra,
            sa.Column("data", JSONB(), nullable=False),
        )

    entity(
        "characters",
        sa.Column("name",                 sa.Text(),    nullable=False),
        sa.Column("char_class",           sa.Text(),    nullable=False, server_default=""),
        sa.Column("level",                sa.Integer(), nullable=False, server_default="1"),
        sa.Column("current_hp",           sa.Integer(), nullable=False, server_default="1"),
        sa.Column("max_hp",               sa.Integer(), nullable=False, server_default="1"),
        sa.Column("ac",                   sa.Integer(), nullable=False, server_default="10"),
        sa.Column("is_player_controlled", sa.Boolean(), nullable=False, server_default="true"),
    )

    entity(
        "monsters",
        sa.Column("name",       sa.Text(),    nullable=False),
        sa.Column("cr",         sa.Text(),    nullable=False, server_default="0"),
        sa.Column("current_hp", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("max_hp",     sa.Integer(), nullable=False, server_default="1"),
        sa.Column("ac",         sa.Integer(), nullable=False, server_default="10"),
    )

    entity(
        "npcs",
        sa.Column("name",          sa.Text(),    nullable=False),
        sa.Column("is_alive",      sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("has_met_party", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("attitude",      sa.Text(),    nullable=False, server_default="indifferent"),
        sa.Column("faction_id",    UUID(as_uuid=False), nullable=True),
    )

    entity(
        "factions",
        sa.Column("name",             sa.Text(),    nullable=False),
        sa.Column("party_reputation", sa.Integer(), nullable=False, server_default="0"),
    )

    entity(
        "quests",
        sa.Column("name",       sa.Text(), nullable=False),
        sa.Column("quest_type", sa.Text(), nullable=False, server_default="side"),
        sa.Column("status",     sa.Text(), nullable=False, server_default="unknown"),
    )

    entity(
        "locations",
        sa.Column("name",      sa.Text(), nullable=False),
        sa.Column("area_type", sa.Text(), nullable=False, server_default="indoor"),
        sa.Column("lighting",  sa.Text(), nullable=False, server_default="bright"),
    )

    entity(
        "containers",
        sa.Column("name",      sa.Text(),    nullable=False),
        sa.Column("is_locked", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_open",   sa.Boolean(), nullable=False, server_default="false"),
    )

    entity(
        "traps",
        sa.Column("name",         sa.Text(),    nullable=False),
        sa.Column("is_detected",  sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_triggered", sa.Boolean(), nullable=False, server_default="false"),
    )

    entity(
        "handouts",
        sa.Column("title",                sa.Text(),    nullable=False),
        sa.Column("handout_type",         sa.Text(),    nullable=False, server_default="other"),
        sa.Column("is_revealed_to_party", sa.Boolean(), nullable=False, server_default="false"),
    )

    entity(
        "sessions",
        sa.Column("session_number", sa.Integer(), nullable=False),
        sa.Column("real_date",      sa.Date(),    nullable=True),
        sa.Column("xp_awarded",     sa.Integer(), nullable=False, server_default="0"),
    )

    entity(
        "encounters",
        sa.Column("is_active",  sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("round",      sa.Integer(), nullable=False, server_default="0"),
        sa.Column("difficulty", sa.Text(),    nullable=False, server_default="medium"),
    )

    # ── Roll log (no data JSONB — all columns are flat) ───────────────────────

    op.create_table(
        "rolls",
        sa.Column("id",          UUID(as_uuid=False), primary_key=True),
        sa.Column("campaign_id", UUID(as_uuid=False),
                  sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("notation",   sa.Text(),               nullable=False),
        sa.Column("result",     sa.Integer(),             nullable=False),
        sa.Column("breakdown",  sa.Text(),               nullable=False),
        sa.Column("rolled_at",  TIMESTAMP(timezone=True), nullable=False),
    )

    # ── Indexes for common access patterns ────────────────────────────────────

    for tbl in (
        "characters", "monsters", "npcs", "factions", "quests",
        "locations", "containers", "traps", "handouts", "sessions",
        "encounters", "rolls",
    ):
        op.create_index(f"ix_{tbl}_campaign_id", tbl, ["campaign_id"])


def downgrade() -> None:
    for tbl in (
        "rolls", "encounters", "sessions", "handouts", "traps",
        "containers", "locations", "quests", "factions", "npcs",
        "monsters", "characters", "campaigns",
    ):
        op.drop_table(tbl)
