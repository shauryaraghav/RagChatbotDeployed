"""
Hybrid retrieval: BM25 (lexical) + dense cosine similarity, min-max
normalized and combined, then re-ranked with a cross-encoder over the
shortlist (cross-encoders score the query and candidate jointly, which is
far more accurate than cosine similarity of independently-embedded
vectors, at the cost of being too slow to run over the whole corpus --
hence "shortlist then re-rank").
"""
import os
import re
import numpy as np
from rank_bm25 import BM25Okapi

_cross_encoder = None  # lazy-loaded, shared across requests


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridIndex:
    def __init__(self, nodes: list):
        self.nodes = nodes
        self._tokenized = [_tokenize(n["text"]) for n in nodes]
        self._bm25 = BM25Okapi(self._tokenized) if nodes else None

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def search(self, query: str, query_embedding, alpha: float = 0.5, top_k: int = 20) -> list:
        """alpha weights dense vs lexical: score = alpha*dense + (1-alpha)*bm25."""
        if not self.nodes:
            return []

        bm25_scores = np.array(self._bm25.get_scores(_tokenize(query)))

        q = np.array(query_embedding)
        q_norm = np.linalg.norm(q) or 1e-9
        dense_scores = np.array([
            float(np.dot(q, np.array(n["embedding"])) /
                  (q_norm * (np.linalg.norm(n["embedding"]) or 1e-9)))
            for n in self.nodes
        ])

        combined = alpha * self._normalize(dense_scores) + (1 - alpha) * self._normalize(bm25_scores)
        order = np.argsort(-combined)[:top_k]

        return [
            {**self.nodes[i], "bm25_score": float(bm25_scores[i]),
             "dense_score": float(dense_scores[i]), "hybrid_score": float(combined[i])}
            for i in order
        ]


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def cross_encoder_rerank(query: str, candidates: list, top_n: int = 7) -> list:
    """Re-score the hybrid shortlist with a cross-encoder and re-sort.
    Falls back to the hybrid order (no crash) if the model can't load,
    e.g. no network access to Hugging Face in a sandboxed environment."""
    if not candidates:
        return []
    if os.getenv("ENABLE_RERANK", "1") == "0":
        # Lite mode: skip the cross-encoder, keep hybrid (BM25 + dense) order.
        for c in candidates:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
        return candidates[:top_n]
    try:
        ce = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        scores = ce.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    except Exception as e:
        for c in candidates:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        candidates[0]["_rerank_error"] = str(e) if candidates else None
    return candidates[:top_n]


def hybrid_search_with_rerank(query, query_embedding, nodes, alpha=0.5, shortlist_k=20, top_n=7) -> list:
    index = HybridIndex(nodes)
    shortlist = index.search(query, query_embedding, alpha=alpha, top_k=shortlist_k)
    return cross_encoder_rerank(query, shortlist, top_n=top_n)
