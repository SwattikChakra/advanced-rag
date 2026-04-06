"""
hybrid.py
─────────
Hybrid retrieval: Dense (FAISS) + Sparse (BM25) fused via Reciprocal Rank Fusion.

Why hybrid?
  Dense retrieval excels at semantic similarity but misses exact keyword matches.
  Sparse retrieval (BM25) excels at keyword matching but misses paraphrases.
  Fusion gives you both — consistently outperforms either alone on BEIR benchmarks.

RRF Formula:
  RRF(d) = Σ 1 / (k + rank_i(d))   where k=60 (empirically optimal, from the original paper)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

from ..indexing.document import Document


# ─────────────────────────────────────────────
# BM25 Sparse Retriever
# ─────────────────────────────────────────────

class BM25Retriever:
    """
    Sparse retriever using BM25Okapi.
    Tokenises by whitespace + lowercasing (extend with NLTK for production).
    """

    def __init__(self, documents: List[Document]) -> None:
        if BM25Okapi is None:
            raise ImportError("pip install rank-bm25")

        self.documents = documents
        tokenised = [self._tokenise(doc.text) for doc in documents]
        self.bm25 = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        tokens = self._tokenise(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.documents[i], float(s)) for i, s in ranked[:top_k]]

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return text.lower().split()


# ─────────────────────────────────────────────
# Dense FAISS Retriever
# ─────────────────────────────────────────────

class DenseRetriever:
    """
    Dense retriever using FAISS IndexFlatIP with L2-normalised vectors.
    Cosine similarity == inner product for normalised vectors.
    """

    def __init__(self, documents: List[Document], embeddings: np.ndarray) -> None:
        if faiss is None:
            raise ImportError("pip install faiss-cpu")
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have equal length")

        self.documents = documents
        self.dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(embeddings.astype(np.float32))

    def search(
        self, query_vector: np.ndarray, top_k: int = 20
    ) -> List[Tuple[Document, float]]:
        qv = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(qv, min(top_k, len(self.documents)))
        return [
            (self.documents[i], float(s))
            for i, s in zip(indices[0], scores[0])
            if i >= 0
        ]


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Merge multiple ranked lists into one via RRF.

    Parameters
    ----------
    ranked_lists : Each list is [(Document, score), ...] sorted by descending relevance.
    k            : RRF constant. 60 is the empirically optimal value from the original paper
                   (Cormack et al., 2009). Higher k reduces the impact of top positions.

    Returns
    -------
    Merged list sorted by descending RRF score.
    """
    rrf_scores: Dict[int, float] = defaultdict(float)
    doc_map: Dict[int, Document] = {}

    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked, start=1):
            doc_id = doc.chunk_id
            rrf_scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in merged]


# ─────────────────────────────────────────────
# Hybrid Retriever (unified API)
# ─────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 sparse retrieval and FAISS dense retrieval via RRF.

    Parameters
    ----------
    documents        : All indexed Document chunks.
    embeddings       : Pre-computed float32 embeddings (N, dim).
    dense_weight     : Not used in RRF directly, but controls candidate pool size ratio.
    rrf_k            : RRF constant (default 60).
    """

    def __init__(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        rrf_k: int = 60,
    ) -> None:
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents, embeddings)

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top_k results by fusing BM25 + dense scores via RRF.

        Parameters
        ----------
        query        : Raw query string (for BM25).
        query_vector : Embedded query (for dense).
        top_k        : Final number of results.
        candidate_k  : How many candidates each retriever fetches before fusion.
        """
        sparse_results = self.bm25.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query_vector, top_k=candidate_k)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results], k=self.rrf_k
        )
        return fused[:top_k]

    def search_dense_only(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Baseline dense-only retrieval for comparison."""
        return self.dense.search(query_vector, top_k=top_k)

    def search_sparse_only(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Baseline sparse-only retrieval for comparison."""
        return self.bm25.search(query, top_k=top_k)
