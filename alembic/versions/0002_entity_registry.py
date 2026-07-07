"""entity registry (Lore Registry — Stage 1)

Additive only: two new tables, no existing table touched.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-06
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lore_entities",
        sa.Column("id",                UUID(as_uuid=False), primary_key=True),
        sa.Column("book_slug",         sa.Text(), nullable=False),
        sa.Column("source_type",       sa.Text(), nullable=False, server_default="adventure"),
        sa.Column("entity_type",       sa.Text(), nullable=False),
        sa.Column("canonical_name",    sa.Text(), nullable=False),
        sa.Column("rolled_up_profile", JSONB(), nullable=False),
        sa.Column("source_chunk_ids",  ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("spoiler_tier",      sa.Text(), nullable=False, server_default="public"),
        sa.Column("created_at",        TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("book_slug", "entity_type", "canonical_name", name="uq_lore_entities_book_type_name"),
    )
    op.create_table(
        "lore_entity_aliases",
        sa.Column("id",              UUID(as_uuid=False), primary_key=True),
        sa.Column("lore_entity_id",  UUID(as_uuid=False),
                  sa.ForeignKey("lore_entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("alias",           sa.Text(), nullable=False),
        sa.UniqueConstraint("lore_entity_id", "alias", name="uq_lore_entity_aliases_entity_alias"),
    )
    op.create_index("ix_lore_entity_aliases_alias_lower", "lore_entity_aliases", [sa.text("lower(alias)")])


def downgrade() -> None:
    op.drop_index("ix_lore_entity_aliases_alias_lower", table_name="lore_entity_aliases")
    op.drop_table("lore_entity_aliases")
    op.drop_table("lore_entities")
