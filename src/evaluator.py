"""
Task 9: Model Evaluation & Tuning

Implements standard IR evaluation metrics:
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- P@K (Precision at K)

Also provides ablation testing and diagnostic tools.
"""

import math
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Compute DCG@K (Discounted Cumulative Gain at K)."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(
    predicted_ids: list[str],
    ground_truth: dict[str, float],
    k: int,
) -> float:
    """
    Compute NDCG@K.

    Args:
        predicted_ids: List of candidate_ids in predicted rank order
        ground_truth: Dict of candidate_id -> relevance score
        k: Cutoff

    Returns:
        NDCG@K score between 0 and 1
    """
    # Get relevance scores for predicted order
    predicted_relevance = [ground_truth.get(cid, 0.0) for cid in predicted_ids[:k]]

    # Get ideal relevance scores (sorted descending)
    all_relevance = sorted(ground_truth.values(), reverse=True)
    ideal_relevance = all_relevance[:k]

    dcg = dcg_at_k(predicted_relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_average_precision(
    predicted_ids: list[str],
    relevant_set: set[str],
) -> float:
    """
    Compute MAP (Mean Average Precision).

    Args:
        predicted_ids: List of candidate_ids in predicted rank order
        relevant_set: Set of candidate_ids that are relevant

    Returns:
        MAP score between 0 and 1
    """
    if not relevant_set:
        return 0.0

    precision_sum = 0.0
    relevant_found = 0

    for i, cid in enumerate(predicted_ids):
        if cid in relevant_set:
            relevant_found += 1
            precision_sum += relevant_found / (i + 1)

    return precision_sum / len(relevant_set)


def precision_at_k(
    predicted_ids: list[str],
    relevant_set: set[str],
    k: int,
) -> float:
    """
    Compute P@K (Precision at K).

    Args:
        predicted_ids: List of candidate_ids in predicted rank order
        relevant_set: Set of candidate_ids that are relevant
        k: Cutoff

    Returns:
        P@K score between 0 and 1
    """
    if k == 0:
        return 0.0

    top_k = predicted_ids[:k]
    relevant_in_top_k = sum(1 for cid in top_k if cid in relevant_set)
    return relevant_in_top_k / k


def compute_composite_score(
    predicted_ids: list[str],
    ground_truth: dict[str, float],
    relevant_set: Optional[set[str]] = None,
) -> dict:
    """
    Compute the full composite score as defined in the competition:
    Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10

    Args:
        predicted_ids: List of candidate_ids in predicted rank order
        ground_truth: Dict of candidate_id -> relevance score
        relevant_set: Optional set of relevant candidate_ids (derived from ground_truth if None)

    Returns:
        Dict with individual metrics and composite score
    """
    if relevant_set is None:
        # Consider all candidates with relevance > 0 as relevant
        relevant_set = {cid for cid, rel in ground_truth.items() if rel > 0}

    n10 = ndcg_at_k(predicted_ids, ground_truth, 10)
    n50 = ndcg_at_k(predicted_ids, ground_truth, 50)
    map_score = mean_average_precision(predicted_ids, relevant_set)
    p10 = precision_at_k(predicted_ids, relevant_set, 10)

    composite = 0.50 * n10 + 0.30 * n50 + 0.15 * map_score + 0.05 * p10

    return {
        "NDCG@10": round(n10, 4),
        "NDCG@50": round(n50, 4),
        "MAP": round(map_score, 4),
        "P@10": round(p10, 4),
        "composite": round(composite, 4),
    }


def analyze_ranking(ranked_candidates: list[dict]) -> dict:
    """
    Analyze the quality of a ranking by looking at feature distributions
    of the top candidates.

    Args:
        ranked_candidates: List of ranked candidate dicts from ranking_engine

    Returns:
        Analysis dict with statistics
    """
    if not ranked_candidates:
        return {}

    top_10 = ranked_candidates[:10]
    top_50 = ranked_candidates[:50]
    top_100 = ranked_candidates[:100]

    def _stats(candidates, feature_key):
        values = [c.get("features", {}).get(feature_key, 0) for c in candidates]
        if not values:
            return {"mean": 0, "min": 0, "max": 0}
        return {
            "mean": round(sum(values) / len(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    analysis = {
        "top_10": {
            "score_range": {
                "min": round(top_10[-1]["score"], 4),
                "max": round(top_10[0]["score"], 4),
            },
            "experience_years": _stats(top_10, "total_experience_years"),
            "ml_experience_years": _stats(top_10, "estimated_ml_experience_years"),
            "must_have_clusters": _stats(top_10, "must_have_clusters_matched"),
            "product_company_ratio": _stats(top_10, "product_company_ratio"),
            "recruiter_response_rate": _stats(top_10, "recruiter_response_rate"),
            "honeypot_score": _stats(top_10, "honeypot_score"),
            "keyword_stuffer_score": _stats(top_10, "keyword_stuffer_score"),
        },
        "top_100": {
            "score_range": {
                "min": round(top_100[-1]["score"], 4) if len(top_100) == 100 else None,
                "max": round(top_100[0]["score"], 4),
            },
            "honeypots": sum(
                1 for c in top_100
                if c.get("features", {}).get("is_likely_honeypot", 0) > 0.5
            ),
            "consulting_only": sum(
                1 for c in top_100
                if c.get("features", {}).get("consulting_only_flag", 0) > 0.5
            ),
            "non_relevant_titles": sum(
                1 for c in top_100
                if c.get("features", {}).get("is_non_relevant_title", 0) > 0.5
            ),
            "india_based": sum(
                1 for c in top_100
                if c.get("features", {}).get("is_india", 0) > 0.5
            ),
            "avg_experience": _stats(top_100, "total_experience_years")["mean"],
            "avg_ml_experience": _stats(top_100, "estimated_ml_experience_years")["mean"],
        },
    }

    # Title distribution in top 100
    title_counts = {}
    for c in top_100:
        # Get title from features isn't available, check candidate_id pattern
        title = c.get("features", {}).get("current_title_relevance", 0)
        score_bucket = "high_relevance" if title >= 0.7 else "medium" if title >= 0.3 else "low"
        title_counts[score_bucket] = title_counts.get(score_bucket, 0) + 1
    analysis["top_100"]["title_relevance_distribution"] = title_counts

    return analysis


def ablation_study(
    ranked_candidates: list[dict],
    all_features: dict,
    similarities: np.ndarray,
    candidate_ids: list[str],
) -> dict:
    """
    Run ablation study: remove each scoring component and measure impact.

    Returns dict with component -> score change.
    """
    from .ranking_engine import compute_component_scores, compute_final_score, DEFAULT_WEIGHTS

    # Baseline: full model
    baseline_scores = []
    for i, cid in enumerate(candidate_ids):
        features = all_features.get(cid, {})
        sem_sim = float(similarities[i])
        components = compute_component_scores(features, sem_sim)
        final = compute_final_score(components, features)
        baseline_scores.append(final)

    baseline_top100 = sorted(
        zip(candidate_ids, baseline_scores),
        key=lambda x: -x[1]
    )[:100]
    baseline_ids = [cid for cid, _ in baseline_top100]

    # Ablation: zero out each component
    results = {}
    for component_name in DEFAULT_WEIGHTS:
        ablated_weights = DEFAULT_WEIGHTS.copy()
        ablated_weights[component_name] = 0.0

        # Renormalize
        total = sum(ablated_weights.values())
        if total > 0:
            for k in ablated_weights:
                ablated_weights[k] /= total

        ablated_scores = []
        for i, cid in enumerate(candidate_ids):
            features = all_features.get(cid, {})
            sem_sim = float(similarities[i])
            components = compute_component_scores(features, sem_sim)
            final = compute_final_score(components, features, ablated_weights)
            ablated_scores.append(final)

        ablated_top100 = sorted(
            zip(candidate_ids, ablated_scores),
            key=lambda x: -x[1]
        )[:100]
        ablated_ids = [cid for cid, _ in ablated_top100]

        # Overlap with baseline
        overlap = len(set(baseline_ids) & set(ablated_ids))
        results[component_name] = {
            "top100_overlap": overlap,
            "top100_change": 100 - overlap,
        }

    return results
