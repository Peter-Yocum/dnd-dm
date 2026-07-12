"""
Reranker — the one interface in Stage 0 that earns a Protocol: two local
implementations are genuinely plausible right now (a downloaded cross-encoder
model vs. a local Ollama LLM-as-judge call), unlike the chunker/embedder,
which have exactly one plausible implementation each. Swap via constructor
injection into RulesStore/HistoryStore — no caller changes.

LLMJudgeReranker is the default (see main.py) — this is a low-throughput
single-user app, so the extra Ollama round-trip's latency is negligible next
to the mechanics/narrator calls already made every turn, and it avoids
needing torch/sentence-transformers in the container at all: confirmed live
that loading the cross-encoder model OOM-killed under Docker Desktop's
default memory allocation, and torch pulls in a genuinely large dependency
chain (including unused NVIDIA CUDA packages on this ARM64/Apple Silicon
Docker setup) for a component this app can just as well run through Ollama,
which every other LLM call in this app already depends on anyway.
CrossEncoderReranker is kept as an opt-in alternative if a future need
justifies the extra dependency weight (e.g. reranking at much higher volume
than a single local table ever will).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from backend.config import settings

if TYPE_CHECKING:
    from backend.stores.rules_store import RuleChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list["RuleChunk"], top_n: int) -> list["RuleChunk"]:
        ...


class CrossEncoderReranker:
    """Opt-in alternative (not the default — see module docstring): a local
    sentence-transformers cross-encoder. No extra Ollama round-trip, faster
    per call, but requires adding sentence-transformers/torch back to
    requirements.txt (removed) and a one-time model download."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded — don't pay the load cost unless search() is actually called

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, chunks: list["RuleChunk"], top_n: int) -> list["RuleChunk"]:
        if not chunks:
            return []
        model = self._get_model()
        pairs = [(query, c.content) for c in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_n]]


class LLMJudgeReranker:
    """Default (see module docstring): batched relevance scoring via a local
    Ollama call. Slower per call than a cross-encoder, but no new dependency
    at all — reuses the Ollama connection every other LLM call already
    needs, keeping the app container light."""

    def __init__(
        self,
        model: str = settings.mechanics_model,
        ollama_base_url: str = settings.ollama_base_url,
    ) -> None:
        self._model = model
        self._ollama_base_url = ollama_base_url

    def rerank(self, query: str, chunks: list["RuleChunk"], top_n: int) -> list["RuleChunk"]:
        if not chunks:
            return []
        from langchain_core.messages import HumanMessage, SystemMessage

        from backend.llm import ollama_chat

        llm = ollama_chat(model=self._model, base_url=self._ollama_base_url)
        listing = "\n".join(f"{i}: {c.content[:300]}" for i, c in enumerate(chunks))
        response = llm.invoke([
            SystemMessage(content=(
                "You rank passages by relevance to a query. Output one line "
                "per passage number, most relevant first, nothing else."
            )),
            HumanMessage(content=f"""Query: {query}

Passages:
{listing}

Output the passage numbers in order of relevance to the query, most \
relevant first, one number per line, nothing else."""),
        ])
        order: list[int] = []
        for line in response.content.splitlines():
            line = line.strip()
            if line.isdigit():
                idx = int(line)
                if 0 <= idx < len(chunks) and idx not in order:
                    order.append(idx)
        # Skip-on-doubt: any chunk the model's response didn't mention (a
        # malformed/partial response) is appended in its original order
        # rather than silently dropped.
        order.extend(i for i in range(len(chunks)) if i not in order)
        return [chunks[i] for i in order[:top_n]]
