"""
Reciprocal Rank Fusion — combines dense (pgvector) + sparse (Postgres full-
text) rankings into one fused ranking. Pure Python, deterministic, no LLM
call — "determinism at the edges" applied to result fusion.

2026-07-12: this module used to also hold BM25Index (in-memory rank_bm25)
and RulesFtsIndex (SQLite FTS5) — both retired when the rules corpus and
session chronicles moved from ChromaDB to Postgres/pgvector + native
tsvector full-text search (see design.md's Tech Stack table). Keyword search
is now just another column/index on the same Postgres tables
(backend/stores/rules_store.py, history_store.py) — no separate index
artifact to keep in sync.
"""


def reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k: int = 60) -> list[str]:
    """Pure Python, deterministic. score(id) = sum(1/(k+rank+1)) across every
    ranked list it appears in. No LLM call."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
