"""Lightweight, deterministic query-intent router for retrieval configuration selection.

Classifies natural language research queries into:
- COMPARATIVE: Queries asking for comparison, contrast, trade-offs, or relationships
  between multiple distinct research papers, systems, or paradigms.
  Triggers document-diversified retrieval (e.g. max_chunks_per_doc=2).
- STANDARD: Factual, explanatory, single-paper, intra-paper, or multi-part questions.
  Retains unconstrained baseline hybrid RRF retrieval (max_chunks_per_doc=None).

Invariants:
1. Zero LLM calls or external network dependencies.
2. Fully deterministic, rule-based matching using regex cues and domain entity recognition.
3. Conservative: Ambiguous or intra-paper queries default to STANDARD to prevent
   premature document eviction and single-paper recall degradation.
4. Zero ground-truth evaluation labels or target paper IDs are used at runtime.
"""

import re
from enum import Enum
from typing import List, Set, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class QueryIntent(str, Enum):
    """Categorical classification of query retrieval intent."""
    STANDARD = "STANDARD"
    COMPARATIVE = "COMPARATIVE"


class QueryIntentResult(BaseModel):
    """Structured classification result describing detected cues, entities, and route."""
    model_config = ConfigDict(extra="ignore")

    query: str = Field(description="Normalized query text evaluated by router")
    intent: QueryIntent = Field(description="Classified query intent ('COMPARATIVE' or 'STANDARD')")
    cues_matched: List[str] = Field(default_factory=list, description="Specific comparative phrasing cues detected")
    entities_matched: List[str] = Field(default_factory=list, description="Distinct research systems/methods detected")
    is_multi_entity: bool = Field(default=False, description="Whether query names >= 2 distinct system families")
    confidence: float = Field(default=1.0, description="Confidence score for classification (1.0 for deterministic rules)")
    reason: str = Field(default="", description="Human-readable rationale for the routing decision")


# ---------------------------------------------------------------------------
# Cues and Domain Entity Dictionaries
# ---------------------------------------------------------------------------

# Explicit comparative phrases and cues
COMPARISON_CUES = [
    # Compare verbs and nouns
    r"\bcompare\b",
    r"\bcompares\b",
    r"\bcompared\s+(?:with|to)\b",
    r"\bcomparing\b",
    r"\bcomparison\b",
    # Difference expressions
    r"\bdifference\s+between\b",
    r"\bdifferences\s+between\b",
    r"\bhow\s+do(?:es)?\s+.+\s+differ\b",
    r"\bdiffer(?:s|ed|ing)?\s+from\b",
    # Versus expressions
    r"\bversus\b",
    r"\bvs\.?\b",
    # Contrast expressions
    r"\bcontrast\s+(?:between|with)?\b",
    r"\bin\s+contrast\s+to\b",
    # Relational comparison across methods
    r"\bhow\s+do(?:es)?\s+.+\s+relate\s+to\b",
    r"\brelation\s+between\b",
]

# Recognizable distinct system / method families in the research corpus
DOMAIN_SYSTEM_FAMILIES: Dict[str, List[str]] = {
    "RAG": [
        r"(?<!self-)(?<!self )\brag\b(?!as)",
        r"\bretrieval[- ]augmented generation\b",
        r"\brag-sequence\b",
        r"\brag-token\b",
    ],
    "DPR": [
        r"\bdpr\b",
        r"\bdense passage retrieval\b",
        r"\bextractive reader\b",
        r"\bbi-encoder\b",
    ],
    "REALM": [
        r"\brealm\b",
        r"\borqa\b",
    ],
    "T5": [
        r"\bt5\b",
        r"\bt5-11b\b",
        r"\bssm\b",
        r"\bsalient span masking\b",
    ],
    "Self-RAG": [
        r"\bself[- ]rag\b",
        r"\badaptive retrieval\b",
        r"\breflection tokens?\b",
        r"\bissup\b",
        r"\bisrel\b",
    ],
    "RAGAS": [
        r"\bragas\b",
        r"\bcontext precision\b",
        r"\bfaithfulness metric\b",
    ],
    "BM25": [
        r"\bbm25\b",
        r"\bsparse retrieval\b",
        r"\bkeyword retrieval\b",
    ],
    "ColBERT": [
        r"\bcolbert\b",
        r"\blate interaction\b",
    ],
}


def extract_matched_cues(query_lower: str) -> List[str]:
    """Find all matching comparative regex cues in the lowercased query."""
    matched = []
    for pat in COMPARISON_CUES:
        m = re.search(pat, query_lower)
        if m:
            matched.append(m.group(0).strip())
    return matched


def extract_matched_entities(query_lower: str) -> List[str]:
    """Find all unique research system families explicitly mentioned in the query."""
    matched = set()
    for sys_name, patterns in DOMAIN_SYSTEM_FAMILIES.items():
        if any(re.search(pat, query_lower) for pat in patterns):
            matched.add(sys_name)
    return sorted(list(matched))


def classify_query_intent(query: str) -> QueryIntentResult:
    """Classify a natural language research query as COMPARATIVE or STANDARD.

    Routing Strategy:
    1. If the query contains explicit comparison cues:
       - If it explicitly names >= 2 distinct system/method families (e.g., 'RAG and DPR',
         'REALM vs DPR', 'Self-RAG and standard RAG'):
         -> Route to COMPARATIVE.
       - If it names only 1 system family (e.g. 'What is the difference between RAG-Sequence
         and RAG-Token?'):
         -> Route to STANDARD. This is an intra-paper comparison whose evidence resides in
            a single source document; applying cross-document diversification would evict
            target pages and harm recall.
       - If no known systems are named:
         - If an explicit multi-entity relational comparison structure is present
           (e.g., 'compare X with Y', 'how do X and Y differ', 'X vs Y'):
           -> Route to COMPARATIVE.
         - Otherwise, intent is ambiguous -> Route to STANDARD.
    2. If the query does not contain comparison cues:
       - Even if 'and' or multiple factual aspects are present (e.g. 'train passage encoder,
         and what negatives are used?'):
         -> Route to STANDARD. Factual/multi-part questions must not be over-diversified.

    Args:
        query: Raw query text.

    Returns:
        QueryIntentResult with intent enum, matched cues, entities, and rationale.
    """
    clean_q = " ".join((query or "").strip().split())
    if not clean_q:
        return QueryIntentResult(
            query="",
            intent=QueryIntent.STANDARD,
            reason="Empty query defaults to STANDARD",
        )

    q_lower = clean_q.lower()
    cues = extract_matched_cues(q_lower)
    entities = extract_matched_entities(q_lower)
    is_multi_entity = len(entities) >= 2

    # Case 1: Comparison cues detected
    if cues:
        if is_multi_entity:
            return QueryIntentResult(
                query=clean_q,
                intent=QueryIntent.COMPARATIVE,
                cues_matched=cues,
                entities_matched=entities,
                is_multi_entity=True,
                reason=(
                    f"Explicit comparative cue ({', '.join(cues[:2])}) detected with multiple "
                    f"distinct system families ({', '.join(entities)})."
                ),
            )

        # Single system named: Intra-paper comparison check
        if len(entities) == 1:
            return QueryIntentResult(
                query=clean_q,
                intent=QueryIntent.STANDARD,
                cues_matched=cues,
                entities_matched=entities,
                is_multi_entity=False,
                reason=(
                    f"Comparative cue present ({cues[0]}), but all terms belong to a single "
                    f"system family ({entities[0]}). Routed to STANDARD to preserve intra-paper recall."
                ),
            )

        # No specific entity family recognized, but strong explicit comparison syntax present
        structural_comparison = bool(
            re.search(r"\bcompare\s+.+\s+(?:and|with|to)\b", q_lower)
            or re.search(r"\bhow\s+do(?:es)?\s+.+\s+and\s+.+\s+(?:compare|differ)\b", q_lower)
            or re.search(r"\bversus\b|\bvs\.?\b", q_lower)
        )
        if structural_comparison:
            return QueryIntentResult(
                query=clean_q,
                intent=QueryIntent.COMPARATIVE,
                cues_matched=cues,
                entities_matched=entities,
                is_multi_entity=False,
                reason=f"Structural cross-method comparison syntax detected ({cues[0]}).",
            )

        # Ambiguous comparison without identifiable distinct methods
        return QueryIntentResult(
            query=clean_q,
            intent=QueryIntent.STANDARD,
            cues_matched=cues,
            entities_matched=entities,
            is_multi_entity=False,
            reason=f"Ambiguous comparison cue ({cues[0]}) without distinct systems; defaulted conservatively to STANDARD.",
        )

    # Case 2: No comparison cues detected
    return QueryIntentResult(
        query=clean_q,
        intent=QueryIntent.STANDARD,
        cues_matched=[],
        entities_matched=entities,
        is_multi_entity=is_multi_entity,
        reason="Factual or multi-part query without explicit comparative intent.",
    )

