"""rules corpus + session chronicles onto pgvector (replaces ChromaDB)

2026-07-12: ChromaDB's own local vector index needed ~2.6GB resident just to
query the 441k-row "rules" collection, independent of any keyword-index
work — see design.md's Tech Stack table for why this app consolidates onto
Postgres rather than adding a second storage engine. `content_tsv` is a
GENERATED ALWAYS AS column (STORED) so keyword search stays in sync with
`content` automatically — no separate rebuild step, unlike the ChromaDB-era
BM25/FTS5 sidecar files this replaces.

Both tables are populated entirely by application code / offline scripts
(scripts/build_index.py for rule_chunks, backend/stores/history_store.py for
session_chronicle_chunks) — this migration only creates the schema.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-12
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "rule_chunks",
        sa.Column("chunk_id",        sa.Text(), primary_key=True),
        sa.Column("book",            sa.Text(), nullable=False),
        sa.Column("section",         sa.Text(), nullable=False),
        sa.Column("source_type",     sa.Text(), nullable=False),
        sa.Column("adventure",       sa.Text(), nullable=False, server_default=""),
        sa.Column("granularity",     sa.Text(), nullable=False),
        sa.Column("parent_chunk_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("sequence_number", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content",         sa.Text(), nullable=False),
        sa.Column("embedding",       Vector(EMBEDDING_DIM), nullable=True),
    )
    # GENERATED ALWAYS AS column — no create_table support for this in
    # SQLAlchemy Core, added via raw DDL immediately after.
    op.execute(
        "ALTER TABLE rule_chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute("CREATE INDEX rule_chunks_embedding_hnsw ON rule_chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX rule_chunks_tsv_gin ON rule_chunks USING GIN (content_tsv)")
    op.create_index("rule_chunks_source_type", "rule_chunks", ["source_type"])
    op.create_index("rule_chunks_adventure", "rule_chunks", ["adventure"])
    op.create_index("rule_chunks_parent_id", "rule_chunks", ["parent_chunk_id"])
    op.create_index("rule_chunks_granularity", "rule_chunks", ["granularity"])

    op.create_table(
        "session_chronicle_chunks",
        sa.Column("chunk_id",       sa.Text(), primary_key=True),
        sa.Column("campaign_id",    UUID(as_uuid=False),
                  sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id",     sa.Text(), nullable=False),
        sa.Column("session_number", sa.Integer(), nullable=False),
        sa.Column("event_index",    sa.Integer(), nullable=False),
        sa.Column("event_type",     sa.Text(), nullable=False),
        sa.Column("content",        sa.Text(), nullable=False),
        sa.Column("embedding",      Vector(EMBEDDING_DIM), nullable=True),
    )
    op.execute(
        "ALTER TABLE session_chronicle_chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute(
        "CREATE INDEX chronicle_embedding_hnsw ON session_chronicle_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX chronicle_tsv_gin ON session_chronicle_chunks USING GIN (content_tsv)")
    op.create_index("chronicle_campaign_id", "session_chronicle_chunks", ["campaign_id"])


def downgrade() -> None:
    op.drop_table("session_chronicle_chunks")
    op.drop_table("rule_chunks")
    # Deliberately not dropping the vector extension — safe/inert to leave
    # installed even if this migration is reverted.
