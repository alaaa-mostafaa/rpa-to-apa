from typing import Dict, List, Tuple
from faf.bayesian_tracker import CATEGORIES
from faf.chroma_case_store import get_chroma_store

def compute_retrieval_prior(similar_cases: List[dict], weight: float = 0.35) -> Dict[str, float]:
    """
    Build a weighted prior from similar past cases.
    Cases with MATCH ground truth verdict are weighted 2x.
    Cases with MISMATCH are weighted 0.5x (still included but less trusted).
    """
    uniform = 1.0 / len(CATEGORIES)
    if not similar_cases:
        return {cat: uniform for cat in CATEGORIES}

    verdict_weights = {
        "MATCH": 2.0,
        "PARTIAL": 1.2,
        "NO_DATA": 0.8,
        "NOT_SCORABLE": 0.8,
        "MISMATCH": 0.4,
    }

    category_weights = {cat: 0.0 for cat in CATEGORIES}
    total_weight = 0.0

    for case in similar_cases:
        verdict = case.get("gt_verdict", "NO_DATA")
        trust = verdict_weights.get(verdict, 0.8)
        sim = case.get("similarity", 0.8)
        effective_weight = sim * trust
        cat = case["category"]
        if cat in category_weights:
            category_weights[cat] += effective_weight
        total_weight += effective_weight

    if total_weight > 0:
        for cat in category_weights:
            category_weights[cat] /= total_weight

    prior = {}
    for cat in CATEGORIES:
        prior[cat] = (1 - weight) * uniform + weight * category_weights[cat]

    total = sum(prior.values())
    return {k: v / total for k, v in prior.items()}

def get_retrieval_prior(
    commit_title: str,
    error_lines: List[str],
    mentioned_files: List[dict],
) -> Tuple[Dict[str, float], List[dict]]:
    """
    Returns (prior_distribution, similar_cases_info).
    """
    store = get_chroma_store()
    similar = store.find_similar_case_records(
        commit_title=commit_title,
        error_lines=error_lines,
        mentioned_files=mentioned_files,
    )
    prior = compute_retrieval_prior(similar)
    return prior, similar
