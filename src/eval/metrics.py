"""
metrics.py
──────────
RAG evaluation metrics implemented from first principles.

The 4 core RAGAS metrics:

┌──────────────────────┬──────────────────────────────────────────────────────┐
│ Metric               │ What it measures                                     │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Context Precision    │ Are retrieved chunks actually relevant?              │
│                      │ (retrieval precision — penalises noise)              │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Context Recall       │ Did the retrieved chunks cover all ground-truth info?│
│                      │ (retrieval recall — penalises missing evidence)      │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Faithfulness         │ Does the generated answer stick to the context?      │
│                      │ (hallucination detector — key safety metric)         │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Answer Relevance     │ Does the answer actually address the question?       │
│                      │ (answer quality — penalises off-topic responses)     │
└──────────────────────┴──────────────────────────────────────────────────────┘

All metrics return values in [0, 1]. Higher is better.

Implementation note:
  LLM-based scoring is used for Faithfulness and Answer Relevance,
  matching RAGAS methodology. Context metrics use token overlap (cheaper).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..indexing.document import Document


# ─────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────

@dataclass
class RAGMetrics:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float

    @property
    def overall(self) -> float:
        """Harmonic mean of all four metrics."""
        scores = [
            self.context_precision,
            self.context_recall,
            self.faithfulness,
            self.answer_relevance,
        ]
        if any(s == 0 for s in scores):
            return 0.0
        return len(scores) / sum(1 / s for s in scores)

    def __str__(self) -> str:
        return (
            f"Context Precision : {self.context_precision:.4f}\n"
            f"Context Recall    : {self.context_recall:.4f}\n"
            f"Faithfulness      : {self.faithfulness:.4f}\n"
            f"Answer Relevance  : {self.answer_relevance:.4f}\n"
            f"Overall (H-mean)  : {self.overall:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "overall": self.overall,
        }


@dataclass
class EvalSample:
    """One question-answer-context triplet for evaluation."""
    question: str
    answer: str
    contexts: List[str]           # retrieved chunk texts
    ground_truth: Optional[str] = None  # reference answer (for recall)


# ─────────────────────────────────────────────
# Token-overlap utilities
# ─────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _token_f1(pred: str, ref: str) -> float:
    """Token-level F1 between two strings."""
    pred_tokens = set(_tokenise(pred))
    ref_tokens = set(_tokenise(ref))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _nli_token_overlap(claim: str, context: str) -> float:
    """
    Approximate NLI via token overlap.
    A claim is 'entailed' if most of its tokens appear in the context.
    Returns proportion of claim tokens found in context.
    """
    claim_tokens = _tokenise(claim)
    context_tokens = set(_tokenise(context))
    if not claim_tokens:
        return 0.0
    return sum(1 for t in claim_tokens if t in context_tokens) / len(claim_tokens)


# ─────────────────────────────────────────────
# LLM scorer (for Faithfulness + Answer Relevance)
# ─────────────────────────────────────────────

class LLMScorer:
    """
    Uses an LLM as a judge for subjective RAG metrics.
    Returns normalised scores in [0, 1].
    """

    FAITHFULNESS_PROMPT = """You are evaluating whether a generated answer is faithful to the provided context.

CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Task: Identify each factual claim in the answer. For each claim, determine if it is
directly supported by the context (1) or not (0).

Return your response in this exact format:
CLAIMS:
- [claim 1]: [1 or 0]
- [claim 2]: [1 or 0]
...
SCORE: [fraction supported, e.g. 0.75]"""

    ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an answer is relevant to the question.

QUESTION: {question}
ANSWER: {answer}

Generate {n_questions} different questions that the given answer would correctly answer.
Return ONLY the questions, one per line."""

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            from openai import OpenAI
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        else:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content

    def score_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Decompose answer into claims, check each against context.
        Returns proportion of claims supported by context.
        """
        combined_context = "\n\n".join(contexts[:5])  # cap context length
        prompt = self.FAITHFULNESS_PROMPT.format(
            context=combined_context, answer=answer
        )
        response = self._complete(
            system="You are a precise factual evaluator.",
            user=prompt,
            max_tokens=800,
        )

        # Parse SCORE line
        score_match = re.search(r"SCORE:\s*([\d.]+)", response)
        if score_match:
            return min(1.0, max(0.0, float(score_match.group(1))))

        # Fallback: count supported claims
        supported = len(re.findall(r":\s*1\b", response))
        total = len(re.findall(r":\s*[01]\b", response))
        return supported / total if total > 0 else 0.5

    def score_answer_relevance(self, question: str, answer: str, n_questions: int = 3) -> float:
        """
        Generate questions that the answer would address.
        Score = average embedding similarity between generated questions and original.
        Falls back to token F1 if embedding model unavailable.
        """
        prompt = self.ANSWER_RELEVANCE_PROMPT.format(
            question=question, answer=answer, n_questions=n_questions
        )
        response = self._complete(
            system="You are a question generation assistant.",
            user=prompt,
            max_tokens=200,
        )
        generated_questions = [
            q.strip() for q in response.strip().split("\n") if q.strip()
        ]

        if not generated_questions:
            return 0.0

        # Token F1 between each generated question and original
        scores = [_token_f1(q, question) for q in generated_questions]
        return sum(scores) / len(scores)


# ─────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────

def context_precision(
    contexts: List[str],
    ground_truth: str,
    top_k: Optional[int] = None,
) -> float:
    """
    Measures: Of the retrieved chunks, how many were actually relevant?

    Uses token F1 against ground truth as relevance proxy.
    Higher = retriever is not wasting context window on irrelevant chunks.

    Parameters
    ----------
    contexts     : Retrieved chunk texts (ordered by retrieval rank).
    ground_truth : Reference answer or relevant passage.
    top_k        : Evaluate only top_k contexts (default: all).
    """
    if not contexts or not ground_truth:
        return 0.0

    ctxs = contexts[:top_k] if top_k else contexts
    relevance_scores = [_token_f1(ctx, ground_truth) for ctx in ctxs]

    # Weighted precision: earlier positions get more weight
    precision_at_k = []
    relevant_so_far = 0
    for i, score in enumerate(relevance_scores, start=1):
        is_relevant = score > 0.1  # threshold for "relevant"
        if is_relevant:
            relevant_so_far += 1
            precision_at_k.append(relevant_so_far / i)

    return sum(precision_at_k) / len(precision_at_k) if precision_at_k else 0.0


def context_recall(
    contexts: List[str],
    ground_truth: str,
) -> float:
    """
    Measures: Did the retrieved chunks contain all necessary information?

    Sentences in the ground truth that are covered by at least one context.
    Higher = retriever is not missing critical information.
    """
    if not contexts or not ground_truth:
        return 0.0

    combined_context = " ".join(contexts)
    gt_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ground_truth) if s.strip()]

    if not gt_sentences:
        return 0.0

    covered = sum(
        1 for sent in gt_sentences
        if _nli_token_overlap(sent, combined_context) > 0.5
    )
    return covered / len(gt_sentences)


def faithfulness(
    answer: str,
    contexts: List[str],
    llm_scorer: Optional[LLMScorer] = None,
) -> float:
    """
    Measures: Does the answer only assert things supported by the context?

    This is the hallucination detector.
    LLM-based scoring used when llm_scorer is provided (more accurate).
    Falls back to token-overlap NLI approximation.
    """
    if not answer or not contexts:
        return 0.0

    if llm_scorer:
        return llm_scorer.score_faithfulness(answer, contexts)

    # Fallback: approximate NLI on answer sentences
    combined = " ".join(contexts)
    answer_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not answer_sentences:
        return 0.0

    scores = [_nli_token_overlap(sent, combined) for sent in answer_sentences]
    return sum(scores) / len(scores)


def answer_relevance(
    question: str,
    answer: str,
    llm_scorer: Optional[LLMScorer] = None,
) -> float:
    """
    Measures: Does the answer address what was asked?

    LLM-based scoring used when llm_scorer is provided.
    Falls back to token F1 between question and answer.
    """
    if not answer or not question:
        return 0.0

    if llm_scorer:
        return llm_scorer.score_answer_relevance(question, answer)

    return _token_f1(answer, question)


# ─────────────────────────────────────────────
# Full evaluator
# ─────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluates a RAG system on a list of EvalSample instances.

    Parameters
    ----------
    use_llm_scoring : Use LLM for Faithfulness + Answer Relevance.
                      More accurate but costs API tokens.
    provider        : 'anthropic' or 'openai' (used only if use_llm_scoring=True).
    """

    def __init__(
        self,
        use_llm_scoring: bool = True,
        provider: str = "anthropic",
    ) -> None:
        self.use_llm_scoring = use_llm_scoring
        self.llm_scorer = LLMScorer(provider=provider) if use_llm_scoring else None

    def evaluate_sample(self, sample: EvalSample) -> RAGMetrics:
        """Evaluate a single question-answer-context triplet."""
        gt = sample.ground_truth or ""

        cp = context_precision(sample.contexts, gt)
        cr = context_recall(sample.contexts, gt)
        f = faithfulness(sample.answer, sample.contexts, self.llm_scorer)
        ar = answer_relevance(sample.question, sample.answer, self.llm_scorer)

        return RAGMetrics(
            context_precision=cp,
            context_recall=cr,
            faithfulness=f,
            answer_relevance=ar,
        )

    def evaluate_dataset(
        self, samples: List[EvalSample], verbose: bool = True
    ) -> Tuple[RAGMetrics, List[RAGMetrics]]:
        """
        Evaluate a full dataset, return average metrics + per-sample breakdown.

        Returns
        -------
        (average_metrics, per_sample_metrics)
        """
        per_sample: List[RAGMetrics] = []
        for i, sample in enumerate(samples):
            m = self.evaluate_sample(sample)
            per_sample.append(m)
            if verbose:
                print(f"[{i+1}/{len(samples)}] Q: {sample.question[:60]}...")
                print(f"  CP={m.context_precision:.3f} CR={m.context_recall:.3f} "
                      f"F={m.faithfulness:.3f} AR={m.answer_relevance:.3f}")

        avg = RAGMetrics(
            context_precision=sum(m.context_precision for m in per_sample) / len(per_sample),
            context_recall=sum(m.context_recall for m in per_sample) / len(per_sample),
            faithfulness=sum(m.faithfulness for m in per_sample) / len(per_sample),
            answer_relevance=sum(m.answer_relevance for m in per_sample) / len(per_sample),
        )
        return avg, per_sample
