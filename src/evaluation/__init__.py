"""Evaluation module for the Research Intelligence Engine."""

from src.evaluation.models import (
    RetrievedChunkEval,
    QuestionEvaluationResult,
    EvaluationSummary,
)
from src.evaluation.ground_truth import (
    GroundTruthTarget,
    load_ground_truth,
    parse_page_references,
)
from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.reporting import save_json_results, generate_markdown_report

__all__ = [
    "RetrievedChunkEval",
    "QuestionEvaluationResult",
    "EvaluationSummary",
    "GroundTruthTarget",
    "load_ground_truth",
    "parse_page_references",
    "RetrievalEvaluator",
    "save_json_results",
    "generate_markdown_report",
]

