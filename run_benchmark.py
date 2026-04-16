"""
run_benchmark.py
────────────────
CLI runner for the 5-config Advanced RAG ablation benchmark.

Run:
    python run_benchmark.py \
        --docs data/sample_docs/ \
        --eval data/benchmark/eval_set.json \
        --provider anthropic

eval_set.json format:
    [
        {
            "question": "What was the revenue in Q3?",
            "ground_truth": "The revenue in Q3 was $4.2B."
        },
        ...
    ]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_eval_set(path: str):
    with open(path) as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Advanced RAG ablation benchmark")
    parser.add_argument("--docs", required=True, help="Directory of PDF/TXT files to index")
    parser.add_argument("--eval", required=True, help="Path to eval_set.json")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="results", help="Where to save benchmark results")
    parser.add_argument("--max-samples", type=int, default=50, help="Max eval samples to run")
    args = parser.parse_args()

    from src.pipeline import AdvancedRAGPipeline, load_and_chunk, EmbeddingModel
    from src.eval.metrics import RAGEvaluator, EvalSample
    from src.eval.benchmark import RAGBenchmark, BenchmarkConfig, STANDARD_CONFIGS

    # Load documents
    docs_dir = Path(args.docs)
    paths = list(docs_dir.glob("**/*.pdf")) + list(docs_dir.glob("**/*.txt"))
    if not paths:
        print(f"No PDF or TXT files found in {args.docs}")
        return

    print(f"Found {len(paths)} document(s) to index.")

    # Load eval set
    raw_eval = load_eval_set(args.eval)
    eval_samples = [
        EvalSample(
            question=item["question"],
            answer="",  # filled during benchmark run
            contexts=[],
            ground_truth=item.get("ground_truth", ""),
        )
        for item in raw_eval[: args.max_samples]
    ]
    print(f"Loaded {len(eval_samples)} eval samples.")

    # Pre-index documents once (shared across all configs)
    print("\nPre-indexing documents...")
    embed_model = EmbeddingModel()
    documents = load_and_chunk(paths)
    import numpy as np
    embeddings = embed_model.encode([d.text for d in documents])
    print(f"Indexed {len(documents)} chunks.")

    # Pipeline factory — builds a pipeline per config reusing the pre-computed index
    def pipeline_factory(config: BenchmarkConfig):
        return AdvancedRAGPipeline(
            documents=documents,
            embed_model=embed_model,
            embeddings=embeddings,
            provider=args.provider,
            use_hybrid=config.use_hybrid,
            use_reranker=config.use_reranker,
            use_hyde=config.use_hyde,
            use_multi_query=config.use_multi_query,
            use_compression=config.use_compression,
            top_k=config.top_k,
            candidate_k=config.candidate_k,
        )

    # Add a run() method adapter that BenchmarkRunner expects
    class PipelineAdapter:
        def __init__(self, pipeline, config):
            self._p = pipeline
            self._c = config

        def run(self, question, config):
            chunks = self._p.retrieve_only(question)
            answer = self._p.query(question)
            contexts = [doc.text for doc, _ in chunks]
            return answer, contexts

    def adapted_factory(config):
        p = pipeline_factory(config)
        return PipelineAdapter(p, config)

    # Run benchmark
    evaluator = RAGEvaluator(use_llm_scoring=True, provider=args.provider)
    benchmark = RAGBenchmark(
        pipeline_factory=adapted_factory,
        evaluator=evaluator,
        output_dir=args.output_dir,
    )
    results = benchmark.run(eval_samples, configs=STANDARD_CONFIGS)

    print(f"\nBenchmark complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
