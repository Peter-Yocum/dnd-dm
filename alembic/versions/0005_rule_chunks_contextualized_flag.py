"""rule_chunks.contextualized flag — resumable recontextualization pass

2026-07-12: tonight's initial rules-corpus reindex (off ChromaDB) was run
with --skip-contextualization for speed, so it could finish the same night
and unblock live play. Re-running the LLM contextualization pass over the
~411k core-book chunks afterward is a long (multi-hour+) job that needs to
be killable and resumable arbitrarily, the same way the initial index build
already is — but the existing resumability check ("does this chunk_id exist
at all?") can't tell a not-yet-contextualized row apart from a fully-done
one. This flag makes that distinction explicit so a --recontextualize pass
only redoes rows that still need it.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-12
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rule_chunks",
        sa.Column("contextualized", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index("rule_chunks_contextualized", "rule_chunks", ["contextualized"])


def downgrade() -> None:
    op.drop_index("rule_chunks_contextualized", table_name="rule_chunks")
    op.drop_column("rule_chunks", "contextualized")
