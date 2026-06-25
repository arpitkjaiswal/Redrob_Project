#!/usr/bin/env python3
"""
Main Entry Point: Intelligent Candidate Discovery & Ranking Engine

Usage:
    # Step 1: Pre-compute (runs outside 5-min window, only needed once)
    python -m src.precompute --candidates ./candidates.jsonl --output ./precomputed

    # Step 2: Rank (runs within 5-min window on CPU, no network)
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

If pre-computed data exists in ./precomputed, the ranking step loads it
and produces results in seconds. If not, it runs the full pipeline
(pre-compute + rank) end-to-end.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PRECOMPUTED_DIR = "./precomputed"


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Senior AI Engineer role"
    )
    parser.add_argument(
        "--candidates", "-c",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--out", "-o",
        default="./submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--precomputed-dir", "-p",
        default=DEFAULT_PRECOMPUTED_DIR,
        help="Directory with pre-computed artifacts",
    )
    parser.add_argument(
        "--no-precompute",
        action="store_true",
        help="Force full pipeline even if precomputed data exists",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=100,
        help="Number of top candidates to return",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print ranking analysis after generation",
    )

    args = parser.parse_args()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("INTELLIGENT CANDIDATE RANKING ENGINE")
    logger.info("=" * 60)

    # ─── Check if pre-computed data exists ───────────────────────────────
    precomputed_exists = (
        os.path.exists(os.path.join(args.precomputed_dir, "features.pkl"))
        and os.path.exists(os.path.join(args.precomputed_dir, "similarities.npy"))
        and os.path.exists(os.path.join(args.precomputed_dir, "candidate_ids.json"))
        and not args.no_precompute
    )

    if not precomputed_exists:
        logger.info("Pre-computed data not found. Running full pipeline...")

        if not os.path.exists(args.candidates):
            logger.error(f"Candidates file not found: {args.candidates}")
            sys.exit(1)

        from src.data_preparation import prepare_all_data

        prepare_all_data(
            candidates_path=args.candidates,
            output_dir=args.precomputed_dir,
            embedding_mode="sentence_transformers",
            populate_sqlite=True,
        )
    else:
        logger.info(f"Loading pre-computed data from {args.precomputed_dir}")

    # ─── Load custom weights if available ────────────────────────────────
    weights = None
    weights_path = os.path.join(args.precomputed_dir, "best_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            weights = json.load(f)
        logger.info(f"Using tuned weights: {weights}")

    # ─── Rank candidates ─────────────────────────────────────────────────
    from src.ranking_engine import rank_candidates
    from src.reasoning_generator import generate_all_reasoning

    ranked = rank_candidates(
        precomputed_dir=args.precomputed_dir,
        weights=weights,
        top_k=args.top_k,
    )

    # ─── Generate reasoning ──────────────────────────────────────────────
    ranked = generate_all_reasoning(ranked, args.precomputed_dir)

    # ─── Write CSV ───────────────────────────────────────────────────────
    write_submission_csv(ranked, args.out)

    # ─── Update SQLite ───────────────────────────────────────────────────
    db_path = os.path.join(args.precomputed_dir, "redrob_candidates.db")
    if os.path.exists(db_path):
        try:
            from src.db_populator import update_rankings
            update_rankings(db_path, ranked)
            logger.info(f"Updated SQLite database rankings at: {db_path}")
        except Exception as e:
            logger.error(f"Failed to update SQLite rankings: {e}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal ranking time: {elapsed:.1f}s")
    logger.info(f"Output written to: {args.out}")

    # ─── Optional analysis ───────────────────────────────────────────────
    if args.analyze:
        from src.evaluator import analyze_ranking
        analysis = analyze_ranking(ranked)
        logger.info("\n" + "=" * 60)
        logger.info("RANKING ANALYSIS")
        logger.info("=" * 60)
        logger.info(json.dumps(analysis, indent=2))


def write_submission_csv(ranked_candidates: list[dict], output_path: str):
    """
    Write the submission CSV in the required format.

    Format: candidate_id,rank,score,reasoning
    - Exactly 100 rows
    - Ranks 1-100
    - Scores non-increasing
    - Ties broken by candidate_id ascending
    """
    # Ensure exactly 100 candidates
    candidates = ranked_candidates[:100]
    if len(candidates) < 100:
        logger.warning(f"Only {len(candidates)} candidates ranked, expected 100")

    # Round scores to 4 decimal places (matching CSV output format)
    for c in candidates:
        c["score"] = round(c["score"], 4)

    # Ensure scores are non-increasing
    for i in range(1, len(candidates)):
        if candidates[i]["score"] > candidates[i - 1]["score"]:
            candidates[i]["score"] = candidates[i - 1]["score"]

    # Ensure ties are broken by candidate_id ascending
    # Compare using the rounded scores to match CSV representation
    i = 0
    while i < len(candidates):
        # Find run of equal scores (using formatted string comparison)
        score_str = f"{candidates[i]['score']:.4f}"
        j = i
        while j < len(candidates) and f"{candidates[j]['score']:.4f}" == score_str:
            j += 1
        # Sort the run by candidate_id
        if j - i > 1:
            candidates[i:j] = sorted(
                candidates[i:j],
                key=lambda x: x["candidate_id"],
            )
        i = j

    # Re-assign ranks after any reordering
    for rank, c in enumerate(candidates, 1):
        c["rank"] = rank

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for c in candidates:
            # Format score to 4 decimal places
            score_str = f"{c['score']:.4f}"
            reasoning = c.get("reasoning", "").replace('"', "'")
            writer.writerow([
                c["candidate_id"],
                c["rank"],
                score_str,
                reasoning,
            ])

    logger.info(f"Wrote {len(candidates)} candidates to {output_path}")


if __name__ == "__main__":
    main()
