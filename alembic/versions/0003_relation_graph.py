"""incremental relation graph (Stage 1.5)

Additive only: one new table, no existing table touched.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-06
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP, UUID

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "entity_relations",
        sa.Column("id",               UUID(as_uuid=False), primary_key=True),
        sa.Column("campaign_id",      UUID(as_uuid=False),
                  sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_type",      sa.Text(), nullable=False),
        sa.Column("source_id",        sa.Text(), nullable=False),
        sa.Column("source_name",      sa.Text(), nullable=False),
        sa.Column("target_type",      sa.Text(), nullable=False),
        sa.Column("target_id",        sa.Text(), nullable=False),
        sa.Column("target_name",      sa.Text(), nullable=False),
        sa.Column("relation",         sa.Text(), nullable=False),
        sa.Column("description",      sa.Text(), nullable=False, server_default=""),
        sa.Column("source_chunk_ids", ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at",       TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("campaign_id", "source_id", "target_id", "relation", name="uq_entity_relations_edge"),
    )


def downgrade() -> None:
    op.drop_table("entity_relations")
