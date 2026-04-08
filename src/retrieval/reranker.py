"""
reranker.py
───────────
Cross-encoder reranking for precision improvement.

Why reranking?
  Bi-encoders (used in dense retrieval) encode query and document *independently*,
  then compare vectors. Fast, but approximate.

  Cross-encoders encode query + document *together*, seeing full interaction.
  Much slower (can't pre-index), but significantly more accurate.

  Production pattern: bi-encoder retrieves top-50, cross-encoder reranks to top-5.
  This gives you speed at scale + accuracy at the final selection.

Models:
  - cross-encoder/ms-marco-MiniLM-L-6-v2   (fast, good — recommended default)
  - cross-encoder/ms-marco-MiniLM-L-12-v2  (slower, better)
  - BAAI/bge-reranker-base                 (strong multilingual)
  - BAAI/bge-reranker-large                (best quality, heavy)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore

from ..indexing.document import Document


DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Reranks a candidate set of (Document, score) pairs using a cross-encoder.

    Parameters
    ----------
    model_name  : HuggingFace cross-encoder model identifier.
    device      : 'cpu', 'cuda', or None (auto).
    batch_size  : Inference batch size.
    score_threshold : If set, filter out results below this score.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: Optional[str] = None,
        batch_size: int = 32,
        score_threshold: Optional[float] = None,
    ) -> None:
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking.\n"
                "pip install sentence-transformers"
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.score_threshold = score_threshold

        print(f"[Reranker] Loading '{model_name}' ...")
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=512,
        )
        print(f"[Reranker] Ready.")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank candidates using cross-encoder scores.

        Parameters
        ----------
        query      : Original user query.
        candidates : List of (Document, bi-encoder-score) from first-stage retrieval.
        top_k      : Return only top_k after reranking. None = return all.

        Returns
        -------
        Reranked list of (Document, cross_encoder_score), descending.
        """
        if not candidates:
            return []

        pairs = [(query, doc.text) for doc, _ in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = sorted(
            zip([doc for doc, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        if self.score_threshold is not None:
            reranked = [(doc, s) for doc, s in reranked if s >= self.score_threshold]

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def score_single(self, query: str, passage: str) -> float:
        """Score a single (query, passage) pair. Useful for debugging."""
        return float(self._model.predict([(query, passage)])[0])


# ─────────────────────────────────────────────
# Contextual Compression
# ─────────────────────────────────────────────

class ContextualCompressor:
    """
    Extracts only the relevant sentences from each retrieved chunk.

    Reduces context window noise — instead of passing a 512-token chunk
    where only 2 sentences are relevant, pass only those sentences.

    Strategy: sentence-level BM25 against the query.
    For production: use a cross-encoder or LLM-based extractor.
    """

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def compress(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Return documents with text replaced by the most relevant sentences only.
        Original documents are not mutated.
        """
        compressed = []
        for doc, score in documents:
            relevant_text = self._extract_relevant_sentences(query, doc.text)
            if not relevant_text.strip():
                relevant_text = doc.text[:300]  # fallback: first 300 chars

            new_doc = Document(
                text=relevant_text,
                source=doc.source,
                page=doc.page,
                chunk_id=doc.chunk_id,
                metadata={**doc.metadata, "compressed": True, "original_length": len(doc.text)},
            )
            compressed.append((new_doc, score))
        return compressed

    def _extract_relevant_sentences(self, query: str, text: str) -> str:
        """Score sentences by BM25 against the query, return top sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= self.max_sentences:
            return text

        query_tokens = set(query.lower().split())

        def sentence_score(sent: str) -> float:
            sent_tokens = sent.lower().split()
            overlap = sum(1 for t in sent_tokens if t in query_tokens)
            return overlap / (len(sent_tokens) + 1e-6)

        scored = sorted(
            enumerate(sentences),
            key=lambda x: sentence_score(x[1]),
            reverse=True,
        )
        top_indices = sorted([i for i, _ in scored[: self.max_sentences]])
        return " ".join(sentences[i] for i in top_indices)
