"""
HistoryStore — ChromaDB collection for session chronicles.

Each ended session is decomposed into per-event documents (one per summary
paragraph, one per key event bullet) and embedded here, so the DM agent can
do semantic search across past events rather than blindly injecting all
history into every context window, and a retrieved fragment isn't a whole
session's worth of unrelated text. Each document is contextualized (a short
blurb naming the campaign/session) before embedding, for the same reason
build_index.py contextualizes book chunks — a bare fragment like "the
merchant agreed to the deal" loses its "which session, which merchant"
grounding once split small.
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma

from backend.config import settings
from backend.llm import ollama_embeddings
from backend.rag.hybrid import BM25Index, reciprocal_rank_fusion
from backend.rag.reranker import LLMJudgeReranker, Reranker
from backend.stores.rules_store import RuleChunk

log = logging.getLogger(__name__)


class HistoryStore:
    COLLECTION = "session_chronicles"

    def __init__(
        self,
        persist_dir: str = settings.chroma_persist_dir,
        ollama_base_url: str = settings.ollama_base_url,
        reranker: Reranker | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._ollama_base_url = ollama_base_url
        self._store: Chroma | None = None
        # See rules_store.py's identical choice — LLMJudgeReranker avoids
        # needing torch/sentence-transformers in the container at all.
        self._reranker: Reranker = reranker if reranker is not None else LLMJudgeReranker()
        self._bm25: BM25Index | None = None
        self._bm25_loaded = False

    def _chroma(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.COLLECTION,
                embedding_function=ollama_embeddings(base_url=self._ollama_base_url),
                persist_directory=self._persist_dir,
            )
        return self._store

    def _bm25_path(self) -> str:
        return str(Path(self._persist_dir).parent / "bm25_history.pkl")

    def _get_bm25(self) -> BM25Index | None:
        if not self._bm25_loaded:
            self._bm25 = BM25Index.load(self._bm25_path())
            self._bm25_loaded = True
        return self._bm25

    _BM25_FETCH_PAGE_SIZE = 5_000  # see build_index.py's identical constant/comment —
    # an unpaginated get() over a large collection hits a real SQLite bound-
    # variable limit ("too many SQL variables"), confirmed live once the
    # rules collection grew past ~250k+ chunks. Unlikely to bite this
    # collection soon (a handful of events per session), but the same fix
    # belongs here for the same reason.

    def _rebuild_bm25(self) -> None:
        """Rebuilt from the full current collection every time a session is
        added — cheap at this collection's scale (a handful of events per
        session across however many campaigns exist), same "always rebuild,
        never trust incremental staleness" discipline as build_index.py."""
        collection = self._chroma()._collection
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        offset = 0
        while True:
            page = collection.get(
                limit=self._BM25_FETCH_PAGE_SIZE, offset=offset, include=["documents", "metadatas"],
            )
            page_ids = page.get("ids", [])
            if not page_ids:
                break
            ids.extend(page_ids)
            documents.extend(page.get("documents", []))
            metadatas.extend(page.get("metadatas", []))
            offset += len(page_ids)
            if len(page_ids) < self._BM25_FETCH_PAGE_SIZE:
                break

        bm25 = BM25Index.build(ids, documents, metadatas)
        bm25.save(self._bm25_path())
        self._bm25 = bm25
        self._bm25_loaded = True

    def add_session(
        self,
        campaign_id: str,
        session_id: str,
        session_number: int,
        text: str,
        key_events: list[str] | None = None,
    ) -> None:
        """Decomposes `text` (the chronicle summary) into paragraphs, plus
        one document per entry in `key_events`, each contextualized and
        embedded separately."""
        from backend.rag.contextualizer import ChunkContextualizer

        key_events = key_events or []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        events = [(p, "summary") for p in paragraphs] + [(e, "key_event") for e in key_events]
        if not events:
            return

        contextualizer = ChunkContextualizer(ollama_base_url=self._ollama_base_url)
        embeddings_fn = ollama_embeddings(base_url=self._ollama_base_url)

        ids, documents, metadatas, embed_texts = [], [], [], []
        for i, (event_text, event_type) in enumerate(events):
            chunk_id = f"{session_id}::{event_type}::{i}"
            blurb = ""
            try:
                blurb = contextualizer.contextualize(
                    event_text, text, f"session {session_number} chronicle",
                )
            except Exception:
                log.exception("Contextualization failed for chronicle chunk %s; embedding without it", chunk_id)
            ids.append(chunk_id)
            documents.append(event_text)
            embed_texts.append(f"{blurb}\n\n{event_text}" if blurb else event_text)
            metadatas.append({
                "campaign_id": campaign_id,
                "session_id": session_id,
                "session_number": session_number,
                "event_index": i,
                "event_type": event_type,
                "chunk_id": chunk_id,
            })

        vectors = embeddings_fn.embed_documents(embed_texts)
        self._chroma()._collection.upsert(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
        self._rebuild_bm25()

    def _hydrate(self, chunk_ids: list[str]) -> dict[str, RuleChunk]:
        if not chunk_ids:
            return {}
        raw = self._chroma()._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        result: dict[str, RuleChunk] = {}
        for cid, doc, meta in zip(raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])):
            meta = meta or {}
            result[cid] = RuleChunk(
                book="",
                section=f"Session {meta.get('session_number', '?')}",
                content=doc or "",
                chunk_id=cid,
            )
        return result

    def search(self, query: str, campaign_id: str, k: int = 3, wide_k: int = 20) -> list[RuleChunk]:
        """Hybrid (dense + BM25) search over this campaign's chronicle
        events, fused via RRF and reranked down to k."""
        try:
            where = {"campaign_id": {"$eq": campaign_id}}
            dense_hits = self._chroma().similarity_search(query, k=wide_k, filter=where)
            dense_ids = [d.metadata.get("chunk_id") for d in dense_hits if d.metadata.get("chunk_id")]

            bm25 = self._get_bm25()
            bm25_ids = [cid for cid, _ in bm25.search(query, k=wide_k, where=where)] if bm25 else []

            fused = reciprocal_rank_fusion([dense_ids, bm25_ids])[:wide_k] if bm25_ids else dense_ids[:wide_k]
            if not fused:
                return []

            hydrated = self._hydrate(fused)
            candidates = [hydrated[cid] for cid in fused if cid in hydrated]
            return self._reranker.rerank(query, candidates, top_n=k)
        except Exception:
            log.exception("HistoryStore.search failed for campaign_id=%s query=%r", campaign_id, query)
            return []
