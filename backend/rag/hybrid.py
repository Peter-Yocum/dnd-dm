"""
Hybrid retrieval helpers: a BM25 index (Chroma has no native hybrid/sparse
search — see rules_store.py's docstring history) and Reciprocal Rank Fusion
to combine BM25 + dense similarity rankings. Both pure Python/CPU, no LLM
call — "determinism at the edges" applied to result fusion.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

# Confirmed live (eval_retrieval.py, 2026-07-08): the old tokenizer was a
# bare `text.lower().split()` — no punctuation stripping, no stopword
# filtering. A natural-language query like "What is the Wizard class like?"
# tokenized to ['what','is','the','wizard','class','like?'] — five
# near-universal filler tokens (one, "like?", couldn't even match anything,
# since the corpus's own naive tokenizer would never emit a token with a
# trailing "?" attached) drowning out the one actually distinctive term.
# BM25Okapi sums a score per query token, so "class" alone (common across
# every "X CLASS FEATURES" section, plus loads of unrelated DMG prose) was
# enough to rank several completely unrelated sections above the real
# "WIZARD CLASS FEATURES" section, which never appeared in the top 30 at
# all. Fix: strip punctuation (regex word-extraction) and drop a small
# stopword list, applied identically to corpus text at build time and to
# queries at search time — they must stay in lockstep or scores become
# meaningless. No new dependency; this is deliberately not a full NLP
# stopword list (e.g. NLTK's), just enough to stop query-filler words from
# competing with content words for score weight.
_STOPWORDS = frozenset("""
    a an the of in on at is are was were be been being to for with and or
    as that this these those it its i you your my me do does did how what
    which who whom tell me about like
""".split())

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


def _matches_where(metadata: dict, where: dict) -> bool:
    """Replicates the small subset of Chroma's `where` filter syntax this
    app actually uses (see rules_store.py's search()/search_adventure_only()):
    $eq, $in, $or, $and, plus a bare {field: value} shorthand."""
    if "$or" in where:
        return any(_matches_where(metadata, clause) for clause in where["$or"])
    if "$and" in where:
        return all(_matches_where(metadata, clause) for clause in where["$and"])
    for field, cond in where.items():
        value = metadata.get(field)
        if isinstance(cond, dict):
            if "$eq" in cond and value != cond["$eq"]:
                return False
            if "$in" in cond and value not in cond["$in"]:
                return False
        elif value != cond:
            return False
    return True


class BM25Index:
    """In-memory rank_bm25 index over a collection's raw chunk text, keyed by
    chunk_id, with the same metadata dict kept alongside each entry so
    search() can apply the same `where` filters RulesStore/HistoryStore
    already use. Built from Chroma's own stored documents at build_index.py
    time and pickled to disk — rebuilt in full on every build_index.py run
    (cheap, pure CPU) so it's never stale relative to Chroma."""

    def __init__(self) -> None:
        self._bm25 = None
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict] = []

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str], metadatas: list[dict]) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        idx = cls()
        idx._chunk_ids = list(chunk_ids)
        idx._metadatas = list(metadatas)
        idx._bm25 = BM25Okapi([_tokenize(t) for t in texts]) if texts else None
        return idx

    def save(self, path: str) -> None:
        """Atomic write (temp file + os.replace) — same pattern as
        ocr_ingest.py's .partial-then-rename, so a kill mid-write never
        leaves a corrupt pickle at the real path."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.partial"
        with open(tmp, "wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "chunk_ids": self._chunk_ids, "metadatas": self._metadatas}, f,
            )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "BM25Index | None":
        if not Path(path).exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._bm25 = data["bm25"]
        idx._chunk_ids = data["chunk_ids"]
        idx._metadatas = data["metadatas"]
        return idx

    def search(self, query: str, k: int, where: dict | None = None) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...], highest score first."""
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        candidates = [
            (self._chunk_ids[i], float(scores[i]))
            for i in range(len(self._chunk_ids))
            if where is None or _matches_where(self._metadatas[i], where)
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:k]


def reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k: int = 60) -> list[str]:
    """Pure Python, deterministic. score(id) = sum(1/(k+rank+1)) across every
    ranked list it appears in. No LLM call."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
