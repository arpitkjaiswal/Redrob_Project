"""
Task 5: Pre-computation Script

Runs OUTSIDE the 5-minute ranking window.
Pre-computes embeddings, features, and caches data for fast ranking.

Usage:
    python -m src.precompute --candidates ./candidates.jsonl --output ./precomputed
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Pre-compute data for ranking")
    parser.add_argument(
        "--candidates", "-c",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl file",
    )
    parser.add_argument(
        "--output", "-o",
        default="./precomputed",
        help="Output directory for pre-computed artifacts",
    )
    parser.add_argument(
        "--embedding-mode", "-e",
        choices=["tfidf", "sentence_transformers"],
        default="sentence_transformers",
        help="Embedding mode (tfidf is faster, sentence_transformers is better)",
    )
    parser.add_argument(
        "--sqlite", "-s",
        action="store_true",
        default=True,
        help="Populate SQLite database for teammate integration",
    )
    parser.add_argument(
        "--tune-weights", "-t",
        action="store_true",
        help="Run weight tuning after pre-computation",
    )

    args = parser.parse_args()

    # Check candidates file exists
    if not os.path.exists(args.candidates):
        logger.error(f"Candidates file not found: {args.candidates}")
        sys.exit(1)

    start = time.time()

    # ─── Pre-compute data ────────────────────────────────────────────────
    from .data_preparation import prepare_all_data

    logger.info("=" * 60)
    logger.info("STARTING PRE-COMPUTATION")
    logger.info("=" * 60)

    result = prepare_all_data(
        candidates_path=args.candidates,
        output_dir=args.output,
        embedding_mode=args.embedding_mode,
        populate_sqlite=args.sqlite,
    )

    logger.info(f"\nPre-computation complete!")
    logger.info(f"  Candidates: {result['num_candidates']}")
    logger.info(f"  Artifacts saved to: {args.output}")

    # ─── Optional: Weight Tuning ─────────────────────────────────────────
    if args.tune_weights:
        logger.info("\n" + "=" * 60)
        logger.info("STARTING WEIGHT TUNING")
        logger.info("=" * 60)

        from .tune_weights import grid_search_weights
        import json

        tune_result = grid_search_weights(
            precomputed_dir=args.output,
            granularity=5,
        )

        # Save best weights
        weights_path = os.path.join(args.output, "best_weights.json")
        with open(weights_path, "w") as f:
            json.dump(tune_result["best_weights"], f, indent=2)

        logger.info(f"Best weights saved to {weights_path}")

    elapsed = time.time() - start
    logger.info(f"\nTotal elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
