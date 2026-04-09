"""
benchmark.py
────────────
Head-to-head benchmarking: Naive RAG vs Advanced RAG.

This is the core differentiator of this project — almost nobody
publishes actual eval numbers comparing RAG configurations publicly.

Compares 5 configurations:
  1. Naive      : Dense only + no reranking + no query transform
  2. Hybrid     : Dense + BM25 + RRF
  3. Reranked   : Dense + cross-encoder rerank
  4. HyDE       : HyDE query transform + Dense
  5. Advanced   : Hybrid + HyDE + Cross-encoder rerank + Contextual compression

Outputs:
  - Console table
  - results/benchmark_results.json
  - results/benchmark_report.md (with charts-ready data)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..eval.metrics import EvalSample, RAGEvaluator, RAGMetrics
from ..indexing.document import Document


@dataclass
class BenchmarkConfig:
    name: str
    use_hybrid: bool = False
    use_reranker: bool = False
    use_hyde: bool = False
    use_multi_query: bool = False
    use_compression: bool = False
    top_k: int = 5
    candidate_k: int = 20


@dataclass
class BenchmarkResult:
    config_name: str
    metrics: RAGMetrics
    avg_latency_ms: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "metrics": self.metrics.to_dict(),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "n_samples": self.n_samples,
        }


STANDARD_CONFIGS = [
    BenchmarkConfig("Naive RAG"),
    BenchmarkConfig("Hybrid Search", use_hybrid=True),
    BenchmarkConfig("Reranked", use_reranker=True),
    BenchmarkConfig("HyDE", use_hyde=True),
    BenchmarkConfig(
        "Advanced RAG",
        use_hybrid=True,
        use_reranker=True,
        use_hyde=True,
        use_compression=True,
    ),
]


class RAGBenchmark:
    """
    Runs multiple RAG configurations against the same eval set and
    produces a comparative results table.

    Parameters
    ----------
    pipeline_factory : Callable that returns a configured pipeline given BenchmarkConfig.
    evaluator        : RAGEvaluator instance.
    output_dir       : Where to save results.
    """

    def __init__(
        self,
        pipeline_factory,
        evaluator: RAGEvaluator,
        output_dir: str = "results",
    ) -> None:
        self.pipeline_factory = pipeline_factory
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        eval_samples: List[EvalSample],
        configs: Optional[List[BenchmarkConfig]] = None,
    ) -> List[BenchmarkResult]:
        """
        Run all configs against all eval samples.

        Returns
        -------
        List of BenchmarkResult, one per config.
        """
        configs = configs or STANDARD_CONFIGS
        all_results: List[BenchmarkResult] = []

        print(f"\n{'='*70}")
        print(f"  RAG BENCHMARK — {len(eval_samples)} samples × {len(configs)} configs")
        print(f"{'='*70}\n")

        for config in configs:
            print(f"\n── Running: {config.name} ──────────────────────────────────")
            pipeline = self.pipeline_factory(config)

            sample_metrics: List[RAGMetrics] = []
            latencies: List[float] = []

            for sample in eval_samples:
                t0 = time.time()
                answer, contexts = pipeline.run(sample.question, config)
                latency_ms = (time.time() - t0) * 1000
                latencies.append(latency_ms)

                # Update sample with actual answer + contexts for eval
                evaluated_sample = EvalSample(
                    question=sample.question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=sample.ground_truth,
                )
                m = self.evaluator.evaluate_sample(evaluated_sample)
                sample_metrics.append(m)

                print(
                    f"  [{len(sample_metrics)}/{len(eval_samples)}] "
                    f"CP={m.context_precision:.3f} CR={m.context_recall:.3f} "
                    f"F={m.faithfulness:.3f} AR={m.answer_relevance:.3f} "
                    f"({latency_ms:.0f}ms)"
                )

            avg_metrics = RAGMetrics(
                context_precision=sum(m.context_precision for m in sample_metrics) / len(sample_metrics),
                context_recall=sum(m.context_recall for m in sample_metrics) / len(sample_metrics),
                faithfulness=sum(m.faithfulness for m in sample_metrics) / len(sample_metrics),
                answer_relevance=sum(m.answer_relevance for m in sample_metrics) / len(sample_metrics),
            )
            avg_latency = sum(latencies) / len(latencies)

            result = BenchmarkResult(
                config_name=config.name,
                metrics=avg_metrics,
                avg_latency_ms=avg_latency,
                n_samples=len(eval_samples),
            )
            all_results.append(result)

            print(f"\n  {config.name} Summary:")
            print(f"  {avg_metrics}")
            print(f"  Avg Latency: {avg_latency:.0f}ms")

        self._save_results(all_results)
        self._print_comparison_table(all_results)
        return all_results

    def _save_results(self, results: List[BenchmarkResult]) -> None:
        path = self.output_dir / "benchmark_results.json"
        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[Benchmark] Results saved to {path}")

        md_path = self.output_dir / "benchmark_report.md"
        self._write_markdown_report(results, md_path)
        print(f"[Benchmark] Markdown report saved to {md_path}")

    def _write_markdown_report(
        self, results: List[BenchmarkResult], path: Path
    ) -> None:
        lines = [
            "# Advanced RAG Benchmark Results",
            "",
            "Head-to-head comparison of RAG configurations on the same evaluation set.",
            "",
            "## Metrics",
            "",
            "| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevance | Overall | Latency (ms) |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in results:
            m = r.metrics
            lines.append(
                f"| **{r.config_name}** "
                f"| {m.context_precision:.4f} "
                f"| {m.context_recall:.4f} "
                f"| {m.faithfulness:.4f} "
                f"| {m.answer_relevance:.4f} "
                f"| {m.overall:.4f} "
                f"| {r.avg_latency_ms:.0f} |"
            )

        # Improvement over naive
        if len(results) >= 2:
            naive = results[0].metrics.overall
            lines += [
                "",
                "## Improvement Over Naive RAG",
                "",
                "| Configuration | Overall Score | Δ vs Naive |",
                "|---|---|---|",
            ]
            for r in results:
                delta = r.metrics.overall - naive
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"| **{r.config_name}** | {r.metrics.overall:.4f} | {sign}{delta:.4f} |"
                )

        lines += [
            "",
            "## Configuration Details",
            "",
            "| Configuration | Hybrid Search | Reranker | HyDE | Compression |",
            "|---|---|---|---|---|",
        ]

        path.write_text("\n".join(lines))

    def _print_comparison_table(self, results: List[BenchmarkResult]) -> None:
        print(f"\n{'='*70}")
        print("  FINAL COMPARISON TABLE")
        print(f"{'='*70}")
        header = f"{'Config':<20} {'Prec':>6} {'Recall':>7} {'Faith':>7} {'AnsRel':>7} {'Overall':>8} {'ms':>6}"
        print(header)
        print("-" * 70)
        for r in results:
            m = r.metrics
            print(
                f"{r.config_name:<20} "
                f"{m.context_precision:>6.4f} "
                f"{m.context_recall:>7.4f} "
                f"{m.faithfulness:>7.4f} "
                f"{m.answer_relevance:>7.4f} "
                f"{m.overall:>8.4f} "
                f"{r.avg_latency_ms:>6.0f}"
            )
        print(f"{'='*70}")
