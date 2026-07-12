"""
HistoryStore — Postgres/pgvector table for session chronicles (replaces the
ChromaDB "session_chronicles" collection, 2026-07-12 migration — see
design.md's Tech Stack table).

Each ended session is decomposed into per-event documents (one per summary
paragraph, one per key event bullet) and embedded here, so the DM agent can
do semantic search across past events rather than blindly injecting all
history into every context window, and a retrieved fragment isn't a whole
session's worth of unrelated text. Each document is contextualized (a short
blurb naming the campaign/session) before embedding, for the same reason
build_index.py contextualizes book chunks — a bare fragment like "the
merchant agreed to the deal" loses its "which session, which merchant"
grounding once split small.

Fully regenerable from Campaign.sessions (JSONB in the campaigns table) —
this table is a derived index, not a source of truth.
"""

import asyncio
import logging

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.llm import ollama_embeddings
from backend.rag.hybrid import reciprocal_rank_fusion
from backend.rag.reranker import LLMJudgeReranker, Reranker
from backend.stores import tables as t
from backend.stores.rules_store import RuleChunk

log = logging.getLogger(__name__)


class HistoryStore:
    def __init__(
        self,
        engine: AsyncEngine,
        ollama_base_url: str | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._engine = engine
        self._ollama_base_url = ollama_base_url
        self._embeddings = ollama_embeddings(base_url=ollama_base_url) if ollama_base_url else ollama_embeddings()
        # See rules_store.py's identical choice — LLMJudgeReranker avoids
        # needing torch/sentence-transformers in the container at all.
        self._reranker: Reranker = reranker if reranker is not None else LLMJudgeReranker()

    async def add_session(
        self,
        campaign_id: str,
        session_id: str,
        session_number: int,
        text: str,
        key_events: list[str] | None = None,
    ) -> None:
        """Decomposes `text` (the chronicle summary) into paragraphs, plus
        one document per entry in `key_events`, each contextualized and
        embedded separately, then upserted into session_chronicle_chunks.
        content_tsv (keyword search) is a generated column — always in sync
        automatically, no separate rebuild step."""
        from backend.rag.contextualizer import ChunkContextualizer

        key_events = key_events or []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        events = [(p, "summary") for p in paragraphs] + [(e, "key_event") for e in key_events]
        if not events:
            return

        contextualizer = ChunkContextualizer(ollama_base_url=self._ollama_base_url)

        rows = []
        for i, (event_text, event_type) in enumerate(events):
            chunk_id = f"{session_id}::{event_type}::{i}"
            blurb = ""
            try:
                blurb = await asyncio.to_thread(
                    contextualizer.contextualize, event_text, text, f"session {session_number} chronicle",
                )
            except Exception:
                log.exception("Contextualization failed for chronicle chunk %s; embedding without it", chunk_id)
            embed_text = f"{blurb}\n\n{event_text}" if blurb else event_text
            vector = await asyncio.to_thread(self._embeddings.embed_query, embed_text)
            rows.append({
                "chunk_id": chunk_id,
                "campaign_id": campaign_id,
                "session_id": session_id,
                "session_number": session_number,
                "event_index": i,
                "event_type": event_type,
                "content": event_text,
                "embedding": vector,
            })

        stmt = pg_insert(t.session_chronicle_chunks).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={
                "content": stmt.excluded.content,
                "embedding": stmt.excluded.embedding,
                "session_number": stmt.excluded.session_number,
                "event_index": stmt.excluded.event_index,
                "event_type": stmt.excluded.event_type,
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def _hydrate(self, chunk_ids: list[str]) -> dict[str, RuleChunk]:
        if not chunk_ids:
            return {}
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(
                    t.session_chronicle_chunks.c.chunk_id, t.session_chronicle_chunks.c.content,
                    t.session_chronicle_chunks.c.session_number,
                ).where(t.session_chronicle_chunks.c.chunk_id.in_(chunk_ids))
            )
            return {
                row.chunk_id: RuleChunk(
                    book="", section=f"Session {row.session_number}", content=row.content, chunk_id=row.chunk_id,
                )
                for row in result
            }

    async def search(self, query: str, campaign_id: str, k: int = 3, wide_k: int = 20) -> list[RuleChunk]:
        """Hybrid (dense + full-text) search over this campaign's chronicle
        events, fused via RRF and reranked down to k."""
        try:
            campaign_filter = t.session_chronicle_chunks.c.campaign_id == campaign_id
            query_vec = await asyncio.to_thread(self._embeddings.embed_query, query)

            async with self._engine.connect() as conn:
                dense_result = await conn.execute(
                    select(t.session_chronicle_chunks.c.chunk_id)
                    .where(campaign_filter)
                    .order_by(t.session_chronicle_chunks.c.embedding.cosine_distance(query_vec))
                    .limit(wide_k)
                )
                dense_ids = [row.chunk_id for row in dense_result]

                tsquery = func.plainto_tsquery("english", query)
                sparse_result = await conn.execute(
                    select(t.session_chronicle_chunks.c.chunk_id)
                    .where(and_(campaign_filter, t.session_chronicle_chunks.c.content_tsv.op("@@")(tsquery)))
                    .order_by(func.ts_rank_cd(t.session_chronicle_chunks.c.content_tsv, tsquery).desc())
                    .limit(wide_k)
                )
                bm25_ids = [row.chunk_id for row in sparse_result]

            fused = reciprocal_rank_fusion([dense_ids, bm25_ids])[:wide_k] if bm25_ids else dense_ids[:wide_k]
            if not fused:
                return []

            hydrated = await self._hydrate(fused)
            candidates = [hydrated[cid] for cid in fused if cid in hydrated]
            return await asyncio.to_thread(self._reranker.rerank, query, candidates, k)
        except Exception:
            log.exception("HistoryStore.search failed for campaign_id=%s query=%r", campaign_id, query)
            return []
