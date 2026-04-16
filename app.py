"""
app.py
──────
Gradio UI for Advanced RAG pipeline.

Lets you toggle each component on/off interactively and see
how the retrieved chunks and answer change in real time.

Run:
    python app.py
    # Open http://localhost:7860
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional

import gradio as gr
from dotenv import load_dotenv

from src.pipeline import AdvancedRAGPipeline

load_dotenv()

_pipeline: Optional[AdvancedRAGPipeline] = None


def _detect_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise EnvironmentError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file.")


def index_files(files, use_hybrid, use_hyde, use_reranker, use_compression):
    global _pipeline
    if not files:
        return "No files uploaded."
    paths = [f.name for f in files]
    try:
        provider = _detect_provider()
        _pipeline = AdvancedRAGPipeline.from_files(
            paths=paths,
            provider=provider,
            use_hybrid=use_hybrid,
            use_hyde=use_hyde,
            use_reranker=use_reranker,
            use_compression=use_compression,
            index_dir=".cache/index",
        )
        config_str = " + ".join(filter(None, [
            "Hybrid" if use_hybrid else "Dense only",
            "HyDE" if use_hyde else None,
            "Reranker" if use_reranker else None,
            "Compression" if use_compression else None,
        ]))
        return (
            f"Indexed {_pipeline.num_chunks} chunks from {len(paths)} file(s)\n"
            f"Config: {config_str}\n"
            f"Provider: {provider}"
        )
    except Exception as e:
        return f"Error: {e}"


def answer_query(question: str):
    if not question.strip():
        yield "Enter a question."
        return
    if _pipeline is None:
        yield "No documents indexed. Upload files first."
        return
    partial = ""
    try:
        for token in _pipeline.stream_query(question):
            partial += token
            yield partial
    except Exception as e:
        yield f"Error: {e}"


def show_chunks(question: str) -> str:
    if not question.strip() or _pipeline is None:
        return "Index documents and enter a question first."
    chunks = _pipeline.retrieve_only(question)
    lines = []
    for i, (doc, score) in enumerate(chunks, 1):
        lines.append(
            f"**[{i}] {doc.source} — p.{doc.page}** (score: {score:.4f})\n"
            f"{doc.text[:300]}{'...' if len(doc.text) > 300 else ''}"
        )
    return "\n\n---\n\n".join(lines) if lines else "No chunks retrieved."


with gr.Blocks(title="Advanced RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Advanced RAG — Component Explorer\nToggle each RAG component and see how retrieval quality changes.")

    with gr.Tab("Query"):
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Upload PDFs / TXTs", file_types=[".pdf", ".txt"], file_count="multiple")
                use_hybrid = gr.Checkbox(label="Hybrid search (BM25 + Dense + RRF)", value=True)
                use_hyde = gr.Checkbox(label="HyDE query transform", value=False)
                use_reranker = gr.Checkbox(label="Cross-encoder reranker", value=True)
                use_compression = gr.Checkbox(label="Contextual compression", value=False)
                index_btn = gr.Button("Index Documents", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=3)
            with gr.Column(scale=2):
                question = gr.Textbox(label="Question", placeholder="Ask something about your documents...")
                ask_btn = gr.Button("Ask", variant="primary")
                answer = gr.Textbox(label="Answer", interactive=False, lines=10)

        index_btn.click(
            fn=index_files,
            inputs=[files, use_hybrid, use_hyde, use_reranker, use_compression],
            outputs=[status],
        )
        ask_btn.click(fn=answer_query, inputs=[question], outputs=[answer])
        question.submit(fn=answer_query, inputs=[question], outputs=[answer])

    with gr.Tab("Retrieval debugger"):
        gr.Markdown("See exactly which chunks are retrieved before generation.")
        debug_q = gr.Textbox(label="Question")
        debug_btn = gr.Button("Show retrieved chunks")
        chunks_out = gr.Markdown()
        debug_btn.click(fn=show_chunks, inputs=[debug_q], outputs=[chunks_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
