"""Ground truth parser and relevance checker for the evaluation dataset."""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class GroundTruthTarget(BaseModel):
    """Ground truth target specification for an evaluation question."""
    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(description="Question ID (e.g. 'Q01')")
    question_text: str = Field(description="Question text")
    question_type: str = Field(description="Question category")
    expected_documents: List[str] = Field(description="Expected document IDs (e.g. ['P1'])")
    expected_pages: Dict[str, List[int]] = Field(
        description="Map of document ID to list of verified 1-based physical page numbers"
    )
    supporting_excerpt: str = Field(default="", description="Verbatim ground-truth excerpt")

    def is_chunk_relevant(self, document_id: str, physical_page_number: int) -> bool:
        """Determine if a chunk matches the annotated document and physical page targets."""
        target_pages = self.expected_pages.get(document_id)
        if target_pages is None:
            return False
        return physical_page_number in target_pages


def parse_page_references(
    page_refs: str,
    default_doc_id: Optional[str] = None,
) -> Dict[str, Set[int]]:
    """Parse annotated page references text into structured map of {doc_id: set(physical_pages)}.

    Handles single pages, comma-separated pages, and ranges (e.g. 'Index 3–7', 'Index 2, 5').
    Converts 0-based physical page indices to 1-based physical page numbers.

    Args:
        page_refs: Raw page_references string from annotations.
        default_doc_id: Optional fallback document ID if no document prefix is matched.

    Returns:
        Dict mapping paper ID (e.g. 'P1') to a set of 1-based physical page numbers.
    """
    targets: Dict[str, Set[int]] = {}
    parts = re.split(r"(P[1-5])(?::|\s+(?=Physical))", page_refs)
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            doc_id = parts[i]
            segment = parts[i + 1]

            # Find 'Physical Page Index' occurrences in segment
            idx_sec = re.findall(r"Physical Page Index\s*([\d\s,–\-—]+)", segment)
            pages: Set[int] = set()
            for sec in idx_sec:
                clean_sec = re.split(r"[\(;]", sec)[0]
                tokens = [t.strip() for t in clean_sec.split(",")]
                for tok in tokens:
                    if "–" in tok or "-" in tok or "—" in tok:
                        r = re.split(r"[–\-—]", tok)
                        if len(r) == 2 and r[0].strip().isdigit() and r[1].strip().isdigit():
                            st, en = int(r[0].strip()), int(r[1].strip())
                            for idx in range(st, en + 1):
                                pages.add(idx + 1)
                    elif tok.isdigit():
                        pages.add(int(tok) + 1)

            if pages:
                targets.setdefault(doc_id, set()).update(pages)
    elif default_doc_id:
        idx_sec = re.findall(r"Physical Page Index\s*([\d\s,–\-—]+)", page_refs)
        pages = set()
        for sec in idx_sec:
            clean_sec = re.split(r"[\(;]", sec)[0]
            tokens = [t.strip() for t in clean_sec.split(",")]
            for tok in tokens:
                if "–" in tok or "-" in tok or "—" in tok:
                    r = re.split(r"[–\-—]", tok)
                    if len(r) == 2 and r[0].strip().isdigit() and r[1].strip().isdigit():
                        st, en = int(r[0].strip()), int(r[1].strip())
                        for idx in range(st, en + 1):
                            pages.add(idx + 1)
                elif tok.isdigit():
                    pages.add(int(tok) + 1)
        if pages:
            targets[default_doc_id] = pages

    return targets


def load_ground_truth(file_path: Path) -> List[GroundTruthTarget]:
    """Load and parse ground truth targets from answerable_questions.json.

    Args:
        file_path: Path to answerable_questions.json.

    Returns:
        List of GroundTruthTarget instances.
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        raw_questions = json.load(f)

    targets: List[GroundTruthTarget] = []
    for item in raw_questions:
        qid = item["id"]
        q_text = item["question"]
        q_type = item.get("type", "Unknown")
        raw_docs = item.get("source_papers", [])

        # Clean document IDs: e.g. 'P3 (and P2)' -> 'P3', 'P2'
        doc_ids: List[str] = []
        for d in raw_docs:
            found_docs = re.findall(r"P[1-5]", str(d))
            for fd in found_docs:
                if fd not in doc_ids:
                    doc_ids.append(fd)

        page_refs = item.get("page_references", "")
        fallback_doc = doc_ids[0] if doc_ids else None
        parsed_pages_set = parse_page_references(page_refs, default_doc_id=fallback_doc)

        # Convert sets to sorted lists
        expected_pages = {k: sorted(list(v)) for k, v in parsed_pages_set.items()}
        excerpt = item.get("supporting_excerpt", "")

        target = GroundTruthTarget(
            question_id=qid,
            question_text=q_text,
            question_type=q_type,
            expected_documents=doc_ids,
            expected_pages=expected_pages,
            supporting_excerpt=excerpt,
        )
        targets.append(target)

    return targets

