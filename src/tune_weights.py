"""
Task 9 (cont): Weight Tuning

Grid search over scoring weights to find optimal combination.
Since there's no ground truth available, we use heuristic quality
metrics to judge ranking quality.
"""

import json
import logging
import os
import time
import pickle
import itertools
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def quality_heuristic(ranked_candidates: list[dict], top_k: int = 100) -> float:
    """
    Compute a quality heuristic for a ranking when no ground truth is available.
    Based on what the JD explicitly says makes a good candidate.

    Higher is better.
    """
    top = ranked_candidates[:top_k]
    if not top:
        return 0.0

    score = 0.0
    n = len(top)

    for i, c in enumerate(top):
        f = c.get("features", {})
        # Position-weighted (top candidates matter more)
        position_weight = 1.0 / (i + 1)

        # Good signals (what the JD wants)
        good = 0.0
        good += 1.0 if f.get("current_title_relevance", 0) >= 0.7 else 0.0
        good += 1.0 if f.get("estimated_ml_experience_years", 0) >= 3 else 0.0
        good += 1.0 if f.get("must_have_clusters_matched", 0) >= 3 else 0.0
        good += 0.5 if f.get("product_company_ratio", 0) >= 0.5 else 0.0
        good += 0.5 if f.get("has_production_signals", 0) > 0.5 else 0.0
        good += 0.5 if f.get("location_match_score", 0) >= 0.5 else 0.0
        good += 0.5 if f.get("recruiter_response_rate", 0) >= 0.3 else 0.0

        # Bad signals (what the JD doesn't want)
        bad = 0.0
        bad += 3.0 if f.get("is_likely_honeypot", 0) > 0.5 else 0.0
        bad += 2.0 if f.get("consulting_only_flag", 0) > 0.5 else 0.0
        bad += 2.0 if f.get("keyword_stuffer_score", 0) > 0.3 else 0.0
        bad += 1.0 if f.get("is_non_relevant_title", 0) > 0.5 else 0.0
        bad += 0.5 if f.get("is_title_chaser", 0) > 0.5 else 0.0

        score += position_weight * (good - bad)

    return score


def grid_search_weights(
    precomputed_dir: str,
    granularity: int = 5,
) -> dict:
    """
    Grid search over scoring weights to maximize ranking quality.

    Args:
        precomputed_dir: Directory with pre-computed artifacts
        granularity: Number of steps per weight dimension (5 = 0.0, 0.25, 0.5, 0.75, 1.0)

    Returns:
        Dict with best weights and score
    """
    from .ranking_engine import compute_component_scores, compute_final_score

    # Load data
    with open(os.path.join(precomputed_dir, "candidate_ids.json"), "r") as f:
        candidate_ids = json.load(f)

    with open(os.path.join(precomputed_dir, "features.pkl"), "rb") as f:
        all_features = pickle.load(f)

    similarities = np.load(os.path.join(precomputed_dir, "similarities.npy"))

    # Pre-compute component scores for all candidates (do once)
    logger.info("Pre-computing component scores for all candidates...")
    all_component_scores = {}
    for i, cid in enumerate(candidate_ids):
        features = all_features.get(cid, {})
        sem_sim = float(similarities[i])
        all_component_scores[cid] = compute_component_scores(features, sem_sim)

    # Generate weight combinations
    component_names = [
        "semantic_similarity", "skills_match", "career_relevance",
        "behavioral_signals", "education_location",
    ]

    steps = np.linspace(0.05, 0.50, granularity)

    best_score = -float("inf")
    best_weights = None
    iterations = 0

    logger.info(f"Starting grid search with granularity={granularity}...")
    start = time.time()

    # Generate combinations that sum to ~1.0
    for combo in itertools.product(steps, repeat=len(component_names)):
        total = sum(combo)
        if total < 0.5:
            continue

        # Normalize to sum to 1
        weights = {
            name: round(w / total, 3)
            for name, w in zip(component_names, combo)
        }

        # Ensure skills and career have significant weight
        if weights["skills_match"] < 0.15 or weights["career_relevance"] < 0.10:
            continue

        # Score all candidates with these weights
        scored = []
        for cid in candidate_ids:
            components = all_component_scores[cid]
            features = all_features.get(cid, {})
            final = compute_final_score(components, features, weights)
            scored.append({"candidate_id": cid, "score": final, "features": features})

        scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

        # Evaluate quality
        q = quality_heuristic(scored, top_k=100)

        if q > best_score:
            best_score = q
            best_weights = weights.copy()

        iterations += 1

    elapsed = time.time() - start
    logger.info(
        f"Grid search complete: {iterations} iterations in {elapsed:.1f}s. "
        f"Best quality score: {best_score:.4f}"
    )
    logger.info(f"Best weights: {best_weights}")

    return {
        "best_weights": best_weights,
        "best_quality_score": best_score,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
    }
