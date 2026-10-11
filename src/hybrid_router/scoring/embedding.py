"""EmbeddingScorer (experimental stub): kNN over past requests.

Embeds the latest user message with a local Ollama embedding model and scores it
as the fraction of its k nearest past requests that escalated. Telemetry doesn't
store prompt text (privacy), so you build the index from your own logs or as you
route (``index.add(vector, escalated)``).

Embedding needs an async call, so use ``await scorer.ascore(req, ctx)``; the sync
``score`` used by the router falls back to the rule scorer until a precomputed
vector is attached to the request (``req.metadata["embedding"]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import httpx2

from hybrid_router.scoring.base import Scorer, ScoreResult, ScoringContext
from hybrid_router.types import CompletionRequest


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class EmbeddingIndex:
    vectors: list[list[float]] = field(default_factory=list)
    escalated: list[bool] = field(default_factory=list)

    def add(self, vector: list[float], escalated: bool) -> None:
        self.vectors.append(vector)
        self.escalated.append(escalated)

    def query(self, vector: list[float], k: int) -> list[tuple[float, bool]]:
        sims = [(cosine(vector, v), e) for v, e in zip(self.vectors, self.escalated, strict=True)]
        return sorted(sims, key=lambda t: -t[0])[:k]

    def __len__(self) -> int:
        return len(self.vectors)


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self._client = httpx2.AsyncClient(base_url=base_url, timeout=30)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post("/api/embed", json={"model": self.model, "input": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]

    async def aclose(self) -> None:
        await self._client.aclose()


class EmbeddingScorer:
    name = "embedding"

    def __init__(
        self,
        index: EmbeddingIndex,
        embedder: OllamaEmbedder,
        fallback: Scorer,
        k: int = 10,
        min_index: int = 50,
    ):
        self.index, self.embedder, self.fallback = index, embedder, fallback
        self.k, self.min_index = k, min_index

    def _from_vector(
        self, vector: list[float], req: CompletionRequest, ctx: ScoringContext
    ) -> ScoreResult:
        if len(self.index) < self.min_index:
            base = self.fallback.score(req, ctx)
            base.reasons.insert(0, f"embedding index too small ({len(self.index)}); rule score")
            return base
        hits = self.index.query(vector, self.k)
        frac = sum(e for _, e in hits) / len(hits)
        reasons = [
            f"{sum(e for _, e in hits)}/{len(hits)} similar past requests escalated "
            f"(top similarity {hits[0][0]:.2f})"
        ]
        return ScoreResult(score=round(frac, 4), reasons=reasons, features={"knn_escalated": frac})

    def score(self, req: CompletionRequest, ctx: ScoringContext) -> ScoreResult:
        vector = req.metadata.get("embedding")
        if isinstance(vector, list) and vector:
            return self._from_vector(vector, req, ctx)
        return self.fallback.score(req, ctx)

    async def ascore(self, req: CompletionRequest, ctx: ScoringContext) -> ScoreResult:
        [vector] = await self.embedder.embed([req.last_user_text()[:8000]])
        return self._from_vector(vector, req, ctx)
