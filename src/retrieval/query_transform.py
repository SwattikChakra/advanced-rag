"""
query_transform.py
──────────────────
Query transformation strategies for improved retrieval.

Three techniques implemented:

1. Multi-Query Generation
   Generate N paraphrases of the original query, retrieve for each,
   merge results via RRF. Handles vocabulary mismatch and ambiguity.

2. HyDE — Hypothetical Document Embeddings (Gao et al., 2022)
   Instead of embedding the query, generate a *hypothetical* answer to the query
   and embed that instead. Bridges the semantic gap between short queries
   and long, detailed documents.
   Paper: https://arxiv.org/abs/2212.10496

3. Step-Back Prompting (Zheng et al., 2023)
   Abstract the specific query to a higher-level concept before retrieving.
   Helps when the specific question requires general background knowledge.
   Paper: https://arxiv.org/abs/2310.06117
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore


# ─────────────────────────────────────────────
# LLM client (shared across transformers)
# ─────────────────────────────────────────────

def _get_llm_client(provider: str = "anthropic"):
    if provider == "anthropic":
        if anthropic is None:
            raise ImportError("pip install anthropic")
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "openai":
        if _OpenAI is None:
            raise ImportError("pip install openai")
        return _OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _complete(client, provider: str, system: str, user: str, max_tokens: int = 512) -> str:
    if provider == "anthropic":
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    else:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


# ─────────────────────────────────────────────
# Multi-Query Generation
# ─────────────────────────────────────────────

class MultiQueryGenerator:
    """
    Generates N semantically diverse paraphrases of the user's query.

    Usage in pipeline:
      queries = generator.generate("What is the refund policy?")
      # → ["What is the return and refund policy?",
      #     "How can I get my money back?",
      #     "What are the terms for product returns?"]

    Each query is used for retrieval independently, results merged via RRF.
    """

    SYSTEM = """You are a query expansion assistant. Given a user question,
generate {n} different phrasings that capture the same information need
from different angles. Each phrasing should:
- Use different vocabulary
- Approach the question from a different angle
- Still answer the same core question

Return ONLY the {n} questions, one per line. No numbering, no explanations."""

    def __init__(self, n: int = 3, provider: str = "anthropic") -> None:
        self.n = n
        self.provider = provider
        self._client = _get_llm_client(provider)

    def generate(self, query: str) -> List[str]:
        """Return n paraphrases of the query (plus the original)."""
        system = self.SYSTEM.format(n=self.n)
        response = _complete(
            self._client, self.provider, system,
            f"Original question: {query}",
            max_tokens=256,
        )
        paraphrases = [q.strip() for q in response.strip().split("\n") if q.strip()]
        # Always include the original query
        all_queries = [query] + paraphrases[: self.n]
        return list(dict.fromkeys(all_queries))  # deduplicate, preserve order


# ─────────────────────────────────────────────
# HyDE — Hypothetical Document Embeddings
# ─────────────────────────────────────────────

class HyDETransformer:
    """
    Implements HyDE (Hypothetical Document Embeddings).

    Instead of embedding "What is the refund policy?",
    generate a hypothetical answer like:
      "The refund policy allows customers to return products within 30 days..."
    and embed *that*. The embedding will be much closer to the actual
    document passage than a short query embedding would be.

    This consistently improves recall on knowledge-intensive tasks.
    """

    SYSTEM = """You are a document synthesis assistant. Given a question,
write a short, factual passage (2-4 sentences) that would directly answer
the question if it appeared in a document. Write it as a factual statement,
not as a question. Be specific and include plausible details.
Do not say "I don't know" — always generate a plausible answer."""

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider
        self._client = _get_llm_client(provider)

    def transform(self, query: str) -> str:
        """
        Generate a hypothetical document passage for the query.

        Returns
        -------
        A synthetic passage to embed instead of (or alongside) the query.
        """
        response = _complete(
            self._client, self.provider, self.SYSTEM,
            f"Question: {query}",
            max_tokens=200,
        )
        return response.strip()

    def transform_batch(self, queries: List[str]) -> List[str]:
        """Transform multiple queries."""
        return [self.transform(q) for q in queries]


# ─────────────────────────────────────────────
# Step-Back Prompting
# ─────────────────────────────────────────────

class StepBackTransformer:
    """
    Abstracts a specific query to a higher-level concept.

    Example:
      Query: "What is the capital gains tax rate for assets held for 3 years in India?"
      Step-back: "What are the capital gains tax rules in India?"

    Retrieval on the step-back query surfaces background knowledge passages
    that provide context for answering the specific question.
    """

    SYSTEM = """You are a query abstraction assistant. Given a specific question,
generate a more general background question whose answer would help
in answering the specific question. The general question should cover
the broader concept or principle underlying the specific question.

Return ONLY the generalised question. No explanation."""

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider
        self._client = _get_llm_client(provider)

    def transform(self, query: str) -> str:
        response = _complete(
            self._client, self.provider, self.SYSTEM,
            f"Specific question: {query}",
            max_tokens=100,
        )
        return response.strip()


# ─────────────────────────────────────────────
# Query Decomposition
# ─────────────────────────────────────────────

class QueryDecomposer:
    """
    Decomposes a complex, multi-hop question into atomic sub-questions.

    Example:
      Query: "Compare the revenue growth of Apple and Microsoft in 2023 and explain
              which company had a better gross margin strategy."
      Sub-queries:
        1. "What was Apple's revenue in 2023?"
        2. "What was Microsoft's revenue in 2023?"
        3. "What was Apple's gross margin in 2023?"
        4. "What was Microsoft's gross margin in 2023?"
    """

    SYSTEM = """You are a question decomposition assistant. Given a complex question
that requires multiple pieces of information to answer, break it into
simple atomic sub-questions that can each be answered independently.

Return ONLY the sub-questions, one per line, no numbering, no explanations.
If the question is already simple (can be answered in one lookup), return just the original question."""

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider
        self._client = _get_llm_client(provider)

    def decompose(self, query: str) -> List[str]:
        response = _complete(
            self._client, self.provider, self.SYSTEM,
            f"Complex question: {query}",
            max_tokens=300,
        )
        sub_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        return sub_queries if sub_queries else [query]
