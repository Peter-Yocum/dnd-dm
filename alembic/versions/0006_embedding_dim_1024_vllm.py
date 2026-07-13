"""embedding columns 768 -> 1024 (vLLM-metal embeddings migration)

2026-07-13: nomic-embed-text (768-dim, via Ollama) is replaced by
mlx-community/Qwen3-Embedding-0.6B-8bit (1024-dim, via vllm-metal --convert
embed) — see vllm-migration-plan.md §7.7. nomic-embed-text's architecture
(NomicBertModel, a BERT-family encoder) cannot be served via vllm-metal at
all — vllm-metal delegates model loading entirely to mlx_lm, which is a
causal-LM-only library, confirmed live by listing every architecture file
in mlx_lm/models (zero encoder models exist there).

This is a breaking dimension change, not a config bump: every existing
embedding in rule_chunks/session_chronicle_chunks was computed at 768
dimensions and is incompatible with a 1024-dim column. A plain
`ALTER COLUMN ... TYPE vector(1024)` against existing 768-dim data doesn't
have anywhere sensible to put the extra 256 dimensions, so this migration
drops and recreates the embedding column (and its HNSW index, built against
the old dimension) on both tables — content/chunk_id/metadata/contextualized
flag are all untouched, only the embedding vector itself is cleared. A full
re-embed (scripts/build_index.py, backend/stores/history_store.py's
add_session for chronicles) is required after this migration — see the
plan's §7.7 for why folding this into the already-in-progress full corpus
re-embed (from the ChromaDB->pgvector move) is the practical path, not a
third separate reindex pass.

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-13
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

OLD_DIM = 768
NEW_DIM = 1024


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS rule_chunks_embedding_hnsw")
    op.drop_column("rule_chunks", "embedding")
    op.add_column("rule_chunks", sa.Column("embedding", Vector(NEW_DIM), nullable=True))
    op.execute("CREATE INDEX rule_chunks_embedding_hnsw ON rule_chunks USING hnsw (embedding vector_cosine_ops)")

    op.execute("DROP INDEX IF EXISTS chronicle_embedding_hnsw")
    op.drop_column("session_chronicle_chunks", "embedding")
    op.add_column("session_chronicle_chunks", sa.Column("embedding", Vector(NEW_DIM), nullable=True))
    op.execute(
        "CREATE INDEX chronicle_embedding_hnsw ON session_chronicle_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS rule_chunks_embedding_hnsw")
    op.drop_column("rule_chunks", "embedding")
    op.add_column("rule_chunks", sa.Column("embedding", Vector(OLD_DIM), nullable=True))
    op.execute("CREATE INDEX rule_chunks_embedding_hnsw ON rule_chunks USING hnsw (embedding vector_cosine_ops)")

    op.execute("DROP INDEX IF EXISTS chronicle_embedding_hnsw")
    op.drop_column("session_chronicle_chunks", "embedding")
    op.add_column("session_chronicle_chunks", sa.Column("embedding", Vector(OLD_DIM), nullable=True))
    op.execute(
        "CREATE INDEX chronicle_embedding_hnsw ON session_chronicle_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
