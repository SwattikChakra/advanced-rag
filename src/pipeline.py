"""
pipeline.py
───────────
Advanced RAG pipeline with modular configuration.

Supports toggling each component on/off for ablation studies:
  - Hybrid search (BM25 + Dense + RRF)
  - HyDE query transformation
  - Multi-query expansion
  - Cross-encoder reranking
  - Contextual compression

This modular design is what enables the benchmark comparisons.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

from .indexing.document import Document
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import ContextualCompressor, CrossEncoderReranker
from .retrieval.query_transform import (
    HyDETransformer,
    MultiQueryGenerator,
    QueryDecomposer,
)


# ─────────────────────────────────────────────
# Embedding model (same as rag-from-scratch)
# ─────────────────────────────────────────────

class EmbeddingModel:
    DEFAULT = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT, device: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedding] Loading {model_name} ...")
        self._model = SentenceTransformer(model_name, device=device)
        self.dim: int = self._model.get_sentence_embedding_dimension()
        self.model_name = model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        prefix = "Represent this passage for retrieval: " if "bge" in self.model_name else ""
        return self._model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        ).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        prefix = "Represent this query for retrieving relevant passages: " if "bge" in self.model_name else ""
        return self._model.encode(
            [prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)


# ─────────────────────────────────────────────
# Document ingestion (minimal, reuses concepts)
# ─────────────────────────────────────────────

def load_and_chunk(
    paths: List[Union[str, Path]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """Load PDFs/TXTs and return Document chunks."""
    import re
    try:
        import fitz
    except ImportError:
        raise ImportError("pip install pymupdf")

    documents = []
    chunk_id = 0

    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            with fitz.open(str(path)) as doc:
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text("text").strip()
                    if not text:
                        continue
                    chunks = _chunk_text(text, chunk_size, chunk_overlap)
                    for chunk in chunks:
                        documents.append(Document(
                            text=chunk, source=path.name,
                            page=page_num, chunk_id=chunk_id,
                        ))
                        chunk_id += 1
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            for chunk in _chunk_text(text, chunk_size, chunk_overlap):
                documents.append(Document(
                    text=chunk, source=path.name,
                    page=-1, chunk_id=chunk_id,
                ))
                chunk_id += 1

    print(f"[Ingestion] {len(documents)} chunks from {len(paths)} file(s)")
    return documents


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 20]


# ─────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────

RAG_SYSTEM = """You are a precise question-answering assistant.
Answer using ONLY the context passages below.
Cite sources as [filename, p.N].
If context is insufficient, say so explicitly — do not hallucinate."""


def generate_answer(
    query: str,
    context_docs: List[Tuple[Document, float]],
    provider: str = "anthropic",
    model: Optional[str] = None,
    stream: bool = False,
) -> Union[str, Iterator[str]]:
    context_str = "\n\n---\n\n".join(
        f"[{i}] {doc.source}, p.{doc.page} (score: {score:.3f})\n{doc.text}"
        for i, (doc, score) in enumerate(context_docs, 1)
    )
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}\n\nANSWER:"

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        mdl = model or "claude-haiku-4-5"
        if stream:
            def _stream():
                with client.messages.stream(
                    model=mdl, max_tokens=1024, system=RAG_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                ) as s:
                    yield from s.text_stream
            return _stream()
        resp = client.messages.create(
            model=mdl, max_tokens=1024, system=RAG_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        mdl = model or "gpt-4o-mini"
        resp = client.chat.completions.create(
            model=mdl, max_tokens=1024,
            messages=[
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            stream=stream,
        )
        if stream:
            return (chunk.choices[0].delta.content or "" for chunk in resp)
        return resp.choices[0].message.content


# ─────────────────────────────────────────────
# Advanced RAG Pipeline
# ─────────────────────────────────────────────

class AdvancedRAGPipeline:
    """
    Modular Advanced RAG pipeline.

    Each component can be toggled independently for ablation studies.
    This is the architecture that powers the benchmark comparisons.
    """

    def __init__(
        self,
        documents: List[Document],
        embed_model: EmbeddingModel,
        embeddings: np.ndarray,
        provider: str = "anthropic",
        llm_model: Optional[str] = None,
        # ── retrieval config ──────────────────
        use_hybrid: bool = True,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        use_reranker: bool = True,
        use_compression: bool = False,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> None:
        self.documents = documents
        self.embed_model = embed_model
        self.provider = provider
        self.llm_model = llm_model
        self.top_k = top_k
        self.candidate_k = candidate_k

        # ── flags ─────────────────────────────
        self.use_hybrid = use_hybrid
        self.use_hyde = use_hyde
        self.use_multi_query = use_multi_query
        self.use_reranker = use_reranker
        self.use_compression = use_compression

        # ── retriever ─────────────────────────
        self.retriever = HybridRetriever(documents, embeddings)

        # ── optional components ───────────────
        self._reranker: Optional[CrossEncoderReranker] = None
        self._compressor: Optional[ContextualCompressor] = None
        self._hyde: Optional[HyDETransformer] = None
        self._multi_query: Optional[MultiQueryGenerator] = None

        if use_reranker:
            self._reranker = CrossEncoderReranker()
        if use_compression:
            self._compressor = ContextualCompressor(max_sentences=4)
        if use_hyde:
            self._hyde = HyDETransformer(provider=provider)
        if use_multi_query:
            self._multi_query = MultiQueryGenerator(n=3, provider=provider)

    # ── public API ─────────────────────────────

    @classmethod
    def from_files(
        cls,
        paths: List[Union[str, Path]],
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        index_dir: Optional[str] = None,
        **kwargs,
    ) -> "AdvancedRAGPipeline":
        """Build pipeline from file paths with optional index caching."""
        import pickle

        embed_model = EmbeddingModel(model_name=embed_model_name)

        if index_dir and Path(index_dir).exists():
            print(f"[Pipeline] Loading cached index from {index_dir}")
            with open(f"{index_dir}/docs.pkl", "rb") as f:
                documents = pickle.load(f)
            embeddings = np.load(f"{index_dir}/embeddings.npy")
        else:
            documents = load_and_chunk(paths)
            print(f"[Pipeline] Embedding {len(documents)} chunks ...")
            embeddings = embed_model.encode([d.text for d in documents])

            if index_dir:
                Path(index_dir).mkdir(parents=True, exist_ok=True)
                np.save(f"{index_dir}/embeddings.npy", embeddings)
                with open(f"{index_dir}/docs.pkl", "wb") as f:
                    pickle.dump(documents, f)
                print(f"[Pipeline] Index cached to {index_dir}")

        return cls(documents=documents, embed_model=embed_model, embeddings=embeddings, **kwargs)

    def query(self, question: str) -> str:
        """Full pipeline: transform → retrieve → rerank → compress → generate."""
        context = self._retrieve(question)
        return generate_answer(question, context, self.provider, self.llm_model)

    def stream_query(self, question: str) -> Iterator[str]:
        context = self._retrieve(question)
        yield from generate_answer(question, context, self.provider, self.llm_model, stream=True)

    def retrieve_only(self, question: str) -> List[Tuple[Document, float]]:
        return self._retrieve(question)

    @property
    def num_chunks(self) -> int:
        return len(self.documents)

    # ── retrieval chain ────────────────────────

    def _retrieve(self, question: str) -> List[Tuple[Document, float]]:
        """Execute the full retrieval chain."""
        queries = [question]
        query_vec = self.embed_model.encode_query(question)

        # Step 1: HyDE — embed a hypothetical answer instead of the question
        if self.use_hyde and self._hyde:
            hypothetical = self._hyde.transform(question)
            query_vec = self.embed_model.encode_query(hypothetical)

        # Step 2: Multi-query expansion + RRF across query variants
        if self.use_multi_query and self._multi_query:
            queries = self._multi_query.generate(question)
            from .retrieval.hybrid import reciprocal_rank_fusion
            all_results = []
            for q in queries:
                qv = self.embed_model.encode_query(q)
                if self.use_hybrid:
                    r = self.retriever.search(q, qv, top_k=self.candidate_k)
                else:
                    r = self.retriever.search_dense_only(qv, top_k=self.candidate_k)
                all_results.append(r)
            candidates = reciprocal_rank_fusion(all_results)[:self.candidate_k]
        elif self.use_hybrid:
            candidates = self.retriever.search(
                question, query_vec, top_k=self.candidate_k
            )
        else:
            candidates = self.retriever.search_dense_only(
                query_vec, top_k=self.candidate_k
            )

        # Step 3: Cross-encoder reranking
        if self.use_reranker and self._reranker:
            candidates = self._reranker.rerank(question, candidates, top_k=self.top_k)
        else:
            candidates = candidates[: self.top_k]

        # Step 4: Contextual compression
        if self.use_compression and self._compressor:
            candidates = self._compressor.compress(question, candidates)

        return candidates
