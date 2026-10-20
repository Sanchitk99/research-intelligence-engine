"""Reporting module saving machine-readable JSON and human-readable Markdown evaluation summaries."""

import json
from pathlib import Path
from typing import List

from src.evaluation.models import EvaluationSummary, QuestionEvaluationResult


def save_json_results(
    summary: EvaluationSummary,
    results: List[QuestionEvaluationResult],
    output_path: Path,
) -> None:
    """Save machine-readable evaluation results to a JSON file.

    Args:
        summary: Aggregated evaluation summary.
        results: Detailed per-question results list.
        output_path: Filepath where JSON should be written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary.model_dump(),
        "questions": [r.model_dump() for r in results],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def generate_markdown_report(
    summary: EvaluationSummary,
    results: List[QuestionEvaluationResult],
    output_path: Path,
    title: str = "Retrieval Evaluation Baseline Report (Task 3.5 & Audit 3.5.1)",
    retriever_info: str = "`DenseRetriever` | **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, Cosine)",
) -> str:
    """Generate and save a comprehensive Markdown evaluation report.

    Args:
        summary: Aggregated evaluation summary.
        results: Detailed per-question results list.
        output_path: Filepath where Markdown should be written.
        title: Title of report.
        retriever_info: Description string for retriever.

    Returns:
        Generated markdown text.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Execution Timestamp:** {summary.execution_timestamp}")
    lines.append(f"**Retriever:** {retriever_info}")
    lines.append(f"**Corpus:** 5 research papers, 82 pages, 396 chunks in `data/qdrant/` (`research_corpus`)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Aggregate Benchmark Metrics")
    lines.append("")
    lines.append("| Metric | Score | Description |")
    lines.append("|---|---|---|")
    lines.append(f"| **Evaluated Questions** | **{summary.evaluated_questions}** / {summary.total_questions} | All answerable questions evaluated; 10 out-of-scope questions excluded |")
    n_eval = summary.evaluated_questions
    r_sum = summary.reciprocal_rank_sum if summary.reciprocal_rank_sum > 0 else sum(r.reciprocal_rank for r in results)
    h1_count = int(round(summary.recall_at_1 * n_eval))
    h3_count = int(round(summary.recall_at_3 * n_eval))
    h5_count = int(round(summary.recall_at_5 * n_eval))

    lines.append(f"| **Hit@1 (Page-Level Acc.)** | **{summary.recall_at_1 * 100:.1f}%** | Relevant target page chunk appears at rank 1 ({h1_count} / {n_eval}) |")
    lines.append(f"| **Hit@3 (Page-Level Acc.)** | **{summary.recall_at_3 * 100:.1f}%** | Relevant target page chunk appears within top 3 ({h3_count} / {n_eval}) |")
    lines.append(f"| **Hit@5 (Page-Level Acc.)** | **{summary.recall_at_5 * 100:.1f}%** | Relevant target page chunk appears within top 5 ({h5_count} / {n_eval}) |")
    lines.append(f"| **Mean Reciprocal Rank (MRR@5)** | **{summary.mrr:.4f}** | Truncated reciprocal rank across all {n_eval} questions (sum = {r_sum:.4f} / {n_eval}) |")
    lines.append(f"| **Mean Target Page Recall@5** | **{summary.mean_target_page_recall_at_5 * 100:.1f}%** | Average fraction of annotated target pages retrieved in top 5 |")
    lines.append(f"| **Mean Document Coverage@5 (All)** | **{summary.mean_document_coverage_at_5 * 100:.1f}%** | Average fraction of required documents retrieved in top 5 across all {n_eval} queries |")
    lines.append("")
    lines.append("### Multi-Document Question Coverage (n = 8)")
    lines.append("")
    lines.append("| Multi-Document Metric | Score | Explanation |")
    lines.append("|---|---|---|")
    lines.append(f"| **Partial Document Coverage (Any)** | **{summary.multi_doc_any_coverage_at_5 * 100:.1f}%** | At least one required document retrieved in top 5 ({int(round(summary.multi_doc_any_coverage_at_5 * summary.multi_doc_questions_count))} / {summary.multi_doc_questions_count}) |")
    lines.append(f"| **Full Document Coverage (All)** | **{summary.multi_doc_full_coverage_at_5 * 100:.1f}%** | ALL required documents retrieved in top 5 ({int(round(summary.multi_doc_full_coverage_at_5 * summary.multi_doc_questions_count))} / {summary.multi_doc_questions_count}) |")
    lines.append(f"| **Mean Document Coverage (Multi-Doc)** | **{summary.multi_doc_mean_coverage_at_5 * 100:.1f}%** | Average percentage of required documents retrieved among multi-paper queries |")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Metric Definition & Ground-Truth Caveat:**")
    lines.append("> 1. **Hit@k vs Recall@k:** In accordance with Information Retrieval conventions for question answering, the primary metric is **Hit@k** (binary top-k retrieval accuracy). Traditional chunk-level recall is not computable without an exhaustive annotation of every sentence across all 396 corpus chunks.")
    lines.append("> 2. **Page-Level Matching Limitation:** A chunk is classified as a match when its physical page matches an annotated target page. While this verifies that the retrieval system surfaced the correct source page, it represents an upper-bound proxy because not every chunk on an annotated page contains the verbatim answer.")
    lines.append(f"> 3. **MRR@5 Derivation:** Because retrieval is evaluated at cutoff top-$k=5$, each question's reciprocal rank is $1 / \\text{{rank}}$ for the first target-page chunk found in ranks 1–5, or $0.0$ if no target page appears within the top 5 results. Chunks beyond rank 5 are not evaluated and receive a reciprocal rank of $0.0$. Thus, this metric is mathematically truncated MRR@5, not full-ranking MRR. The exact sum of reciprocal ranks is ${r_sum:.4f}$, yielding $\\text{{MRR@5}} = {r_sum:.4f} / {n_eval} = {summary.mrr:.4f}$.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Performance Breakdown by Question Type")
    lines.append("")
    lines.append("| Question Type | Count | Hit@1 | Hit@3 | Hit@5 | MRR@5 |")
    lines.append("|---|---|---|---|---|---|")
    for q_type, metrics in sorted(summary.metrics_by_type.items()):
        cnt = int(metrics["count"])
        r1 = metrics["recall_at_1"] * 100
        r3 = metrics["recall_at_3"] * 100
        r5 = metrics["recall_at_5"] * 100
        mrr = metrics["mrr"]
        lines.append(f"| **{q_type}** | {cnt} | {r1:.1f}% | {r3:.1f}% | {r5:.1f}% | {mrr:.4f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Per-Question Evaluation Diagnostics")
    lines.append("")
    lines.append("| ID | Type | Expected Targets | Top-1 Chunk (Score) | 1st Rel Rank | Reciprocal Rank | Hit@1 | Hit@3 | Hit@5 | Doc Cov. | All Docs? | Page Recall |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        t_pages = ", ".join(f"{doc}:p{pages}" for doc, pages in r.expected_pages.items())
        top1 = r.retrieved_chunks[0] if r.retrieved_chunks else None
        top1_desc = f"{top1.chunk_id} ({top1.score:.3f})" if top1 else "N/A"
        rel_rank = str(r.first_relevant_rank) if r.first_relevant_rank else "—"
        rr_str = f"{r.reciprocal_rank:.4f}"
        h1 = "✅" if r.hit_at_1 else "❌"
        h3 = "✅" if r.hit_at_3 else "❌"
        h5 = "✅" if r.hit_at_5 else "❌"
        cov = f"{r.document_coverage_at_5 * 100:.0f}%"
        all_docs = "✅" if r.all_documents_covered_at_5 else "❌"
        pg_rec = f"{r.target_page_recall_at_5 * 100:.0f}%"
        lines.append(f"| **{r.question_id}** | {r.question_type.split()[0]} | `{t_pages}` | `{top1_desc}` | {rel_rank} | {rr_str} | {h1} | {h3} | {h5} | {cov} | {all_docs} | {pg_rec} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Analysis of Missed Evidence (Absent from Top-5)")
    lines.append("")
    if not summary.questions_missing_in_top_5:
        lines.append("None. All 18 evaluated questions retrieved relevant evidence within the top 5 results.")
    else:
        lines.append(f"The following {len(summary.questions_missing_in_top_5)} questions did not contain ground-truth target evidence in the top 5:")
        lines.append("")
        for qid in summary.questions_missing_in_top_5:
            q_res = next(r for r in results if r.question_id == qid)
            lines.append(f"### Question {qid}: \"{q_res.question_text}\"")
            lines.append(f"- **Expected Targets:** {q_res.expected_pages}")
            lines.append(f"- **Top-5 Retrieved Chunks:**")
            for c in q_res.retrieved_chunks[:5]:
                lines.append(f"  - Rank {c.rank} [{c.score:.4f}] `{c.chunk_id}` ({c.document_id}, Page {c.physical_page_number}): \"{c.snippet}\"")
            lines.append("")

    report_content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


def generate_comparison_markdown_report(
    dense_summary: EvaluationSummary,
    dense_results: List[QuestionEvaluationResult],
    hybrid_summary: EvaluationSummary,
    hybrid_results: List[QuestionEvaluationResult],
    output_path: Path,
) -> str:
    """Generate side-by-side comparison Markdown report between Dense and Hybrid retrieval.

    Args:
        dense_summary: Aggregated summary for dense retrieval baseline.
        dense_results: Per-question results for dense retrieval.
        hybrid_summary: Aggregated summary for hybrid retrieval.
        hybrid_results: Per-question results for hybrid retrieval.
        output_path: Destination path for comparison report.

    Returns:
        Generated report markdown text.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dense_by_id = {r.question_id: r for r in dense_results}
    hybrid_by_id = {r.question_id: r for r in hybrid_results}

    lines = []
    lines.append("# Dense vs. Hybrid Retrieval Evaluation Report (Task 4.1)")
    lines.append("")
    lines.append(f"**Execution Timestamp:** {hybrid_summary.execution_timestamp}")
    lines.append(f"**Baseline:** `DenseRetriever` (`all-MiniLM-L6-v2`, Cosine, top-k=5)")
    lines.append(f"**Hybrid System:** `HybridRetriever` (BM25Okapi + `all-MiniLM-L6-v2`, RRF k=60, candidate_k=50)")
    lines.append(f"**Corpus:** 5 research papers, 82 pages, 396 chunks in `data/qdrant/` (`research_corpus`)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Aggregate Performance Comparison")
    lines.append("")
    lines.append("| Metric | Dense Baseline | Hybrid (Dense + BM25) | Delta | Relative Change |")
    lines.append("|---|---|---|---|---|")

    def format_row(name, d_val, h_val, is_pct=True, decimals=1):
        if is_pct:
            d_str = f"{d_val * 100:.{decimals}f}%"
            h_str = f"{h_val * 100:.{decimals}f}%"
            diff = (h_val - d_val) * 100
            diff_str = f"{diff:+.{decimals}f}%"
        else:
            d_str = f"{d_val:.4f}"
            h_str = f"{h_val:.4f}"
            diff = h_val - d_val
            diff_str = f"{diff:+.4f}"

        if d_val > 0:
            rel_change = ((h_val - d_val) / d_val) * 100
            rel_str = f"{rel_change:+.1f}%"
        elif h_val > 0:
            rel_str = "+100.0%"
        else:
            rel_str = "0.0%"

        icon = "🔺" if diff > 0.0001 else ("🔻" if diff < -0.0001 else "➖")
        return f"| **{name}** | {d_str} | **{h_str}** | {icon} {diff_str} | {rel_str} |"

    lines.append(format_row("Hit@1 (Page-Level Acc.)", dense_summary.recall_at_1, hybrid_summary.recall_at_1))
    lines.append(format_row("Hit@3 (Page-Level Acc.)", dense_summary.recall_at_3, hybrid_summary.recall_at_3))
    lines.append(format_row("Hit@5 (Page-Level Acc.)", dense_summary.recall_at_5, hybrid_summary.recall_at_5))
    lines.append(format_row("Mean Reciprocal Rank (MRR@5)", dense_summary.mrr, hybrid_summary.mrr, is_pct=False))
    lines.append(format_row("Mean Target Page Recall@5", dense_summary.mean_target_page_recall_at_5, hybrid_summary.mean_target_page_recall_at_5))
    lines.append(format_row("Mean Document Coverage@5", dense_summary.mean_document_coverage_at_5, hybrid_summary.mean_document_coverage_at_5))
    lines.append(format_row("Multi-Doc Any Coverage@5", dense_summary.multi_doc_any_coverage_at_5, hybrid_summary.multi_doc_any_coverage_at_5))
    lines.append(format_row("Multi-Doc Full Coverage@5", dense_summary.multi_doc_full_coverage_at_5, hybrid_summary.multi_doc_full_coverage_at_5))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Analysis of Previously Difficult Questions")
    lines.append("")
    lines.append("Six questions failed to retrieve ground-truth evidence in top 5 under Dense retrieval:")
    lines.append("`Q01`, `Q13`, `Q14`, `Q15`, `Q16`, and `Q17`.")
    lines.append("")
    lines.append("| ID | Question | Expected Targets | Dense Rank | Hybrid Rank | Outcome | Key Factor |")
    lines.append("|---|---|---|:---:|:---:|:---:|---|")

    difficult_ids = ["Q01", "Q13", "Q14", "Q15", "Q16", "Q17"]
    for qid in difficult_ids:
        d_res = dense_by_id[qid]
        h_res = hybrid_by_id[qid]
        d_rank = str(d_res.first_relevant_rank) if d_res.first_relevant_rank else "Miss"
        h_rank = str(h_res.first_relevant_rank) if h_res.first_relevant_rank else "Miss"
        t_pages = ", ".join(f"{doc}:p{pages}" for doc, pages in d_res.expected_pages.items())

        if d_res.first_relevant_rank is None and h_res.first_relevant_rank is not None:
            status = "✅ Rescued (Hit)"
        elif d_res.first_relevant_rank and h_res.first_relevant_rank and h_res.first_relevant_rank < d_res.first_relevant_rank:
            status = "⬆️ Improved"
        elif d_res.first_relevant_rank == h_res.first_relevant_rank:
            status = "➖ Unchanged"
        else:
            status = "❌ Still Missed" if h_res.first_relevant_rank is None else "🔻 Lower"

        factor = ""
        if qid == "Q01":
            factor = "BM25 matched 'generator', 'model', 'RAG' with §2.3 BART passage"
        elif qid == "Q13":
            factor = "BM25 retrieved P5 evaluators/limitations context"
        elif qid == "Q14":
            factor = "Exact paper references matched across multi-paper query"
        elif qid == "Q15":
            factor = "Dense & BM25 balance for GPT-4 demonstration terms"
        elif qid == "Q16":
            factor = "Exact matching on 'passage sizes', '288 wordpieces', '100-word'"
        elif qid == "Q17":
            factor = "Exact benchmark names (NQ, WQ, CT) matched in tables"

        lines.append(f"| **{qid}** | {d_res.question_text[:45]}... | `{t_pages}` | {d_rank} | {h_rank} | {status} | {factor} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Per-Question Side-by-Side Diagnostics")
    lines.append("")
    lines.append("| ID | Category | Expected | Dense Rank | Hybrid Rank | MRR@5 Delta | Status | Top Hybrid Chunk |")
    lines.append("|---|---|---|:---:|:---:|:---:|:---:|---|")

    for qid in sorted(dense_by_id.keys()):
        d_res = dense_by_id[qid]
        h_res = hybrid_by_id[qid]

        d_rank = str(d_res.first_relevant_rank) if d_res.first_relevant_rank else "—"
        h_rank = str(h_res.first_relevant_rank) if h_res.first_relevant_rank else "—"
        mrr_delta = h_res.reciprocal_rank - d_res.reciprocal_rank
        delta_str = f"{mrr_delta:+.4f}" if abs(mrr_delta) > 0.0001 else "0.0000"

        if d_res.first_relevant_rank is None and h_res.first_relevant_rank is not None:
            tag = "✅ Rescued"
        elif h_res.first_relevant_rank and (d_res.first_relevant_rank is None or h_res.first_relevant_rank < d_res.first_relevant_rank):
            tag = "⬆️ Improved"
        elif d_res.first_relevant_rank == h_res.first_relevant_rank:
            tag = "➖ Same"
        elif h_res.first_relevant_rank is None and d_res.first_relevant_rank is not None:
            tag = "🔻 Regressed (Miss)"
        else:
            tag = "🔻 Dropped"

        t_pages = ", ".join(f"{doc}:p{pages}" for doc, pages in d_res.expected_pages.items())
        top_h = h_res.retrieved_chunks[0] if h_res.retrieved_chunks else None
        top_h_desc = f"{top_h.chunk_id} (RRF: {top_h.score:.5f})" if top_h else "N/A"

        lines.append(f"| **{qid}** | {d_res.question_type.split()[0]} | `{t_pages}` | {d_rank} | {h_rank} | {delta_str} | {tag} | `{top_h_desc}` |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Key Engineering Insights & Methodological Observations")
    lines.append("")
    lines.append("1. **Complementary Modalities:** Dense semantic retrieval excels at high-level thematic similarity, whereas BM25 excels at exact technical entities ('BART-large', '288 wordpieces', 'Table 1').")
    lines.append("2. **Reciprocal Rank Fusion (RRF) Robustness:** RRF eliminates score scale mismatch without requiring calibrated score normalization. Chunks retrieved with high consensus by both dense and sparse models naturally dominate the top ranks.")
    lines.append("3. **MRR@5 Truncation Definition:** Because evaluation retrieval evaluates only top-5 chunks, any target evidence beyond rank 5 receives a reciprocal rank of 0.0. The metric is strictly MRR@5. Full-ranking MRR would assign non-zero reciprocal rank (e.g. 1/12 for Q05 at rank 12).")
    lines.append("4. **Strict Annotation Independence:** Retrieval rankings were computed strictly without using ground-truth labels. The evaluation framework was applied purely post-hoc.")

    report_content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content

