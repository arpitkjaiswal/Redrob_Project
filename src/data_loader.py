"""
Task 1: Data Collection & Loading

Efficiently loads 100K candidate profiles from candidates.jsonl (487MB).
Uses streaming/line-by-line parsing to stay within 16GB RAM constraint.
"""

import json
import gzip
import os
import time
import logging
from pathlib import Path
from typing import Iterator

from .schema import Candidate

logger = logging.getLogger(__name__)


def load_candidates_stream(filepath: str) -> Iterator[dict]:
    """
    Stream-load candidates line by line from JSONL (or gzipped JSONL).
    Yields raw dicts to minimize memory usage during initial load.
    """
    path = Path(filepath)

    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")

    line_count = 0
    error_count = 0

    with opener() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                line_count += 1
                yield data
            except json.JSONDecodeError as e:
                error_count += 1
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")

    logger.info(f"Loaded {line_count} candidates ({error_count} errors)")


def load_all_candidates(filepath: str) -> list[Candidate]:
    """
    Load all candidates into memory as structured Candidate objects.
    For 100K candidates, this uses ~2-4GB RAM.
    """
    start = time.time()
    candidates = []

    for data in load_candidates_stream(filepath):
        try:
            candidate = Candidate.from_dict(data)
            candidates.append(candidate)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Skipping candidate {data.get('candidate_id', '?')}: {e}")

    elapsed = time.time() - start
    logger.info(f"Parsed {len(candidates)} candidates in {elapsed:.1f}s")
    return candidates


def load_all_candidates_raw(filepath: str) -> list[dict]:
    """
    Load all candidates as raw dicts (lighter memory, faster).
    Used when we need to process features without full object overhead.
    """
    start = time.time()
    candidates = list(load_candidates_stream(filepath))
    elapsed = time.time() - start
    logger.info(f"Loaded {len(candidates)} raw candidate dicts in {elapsed:.1f}s")
    return candidates


def load_job_description() -> dict:
    """
    Returns the structured job description for the Senior AI Engineer role.
    Hard-coded from the JD document since we need to deeply understand
    what the JD *means*, not just what it *says*.
    """
    return {
        "title": "Senior AI Engineer — Founding Team",
        "company": "Redrob AI",
        "type": "Series A AI-native talent intelligence platform",
        "location": {
            "preferred": ["Pune", "Noida"],
            "acceptable": ["Hyderabad", "Mumbai", "Delhi NCR", "Bangalore", "Bengaluru"],
            "country": "India",
            "work_mode": "hybrid",
            "relocation_ok": True,
            "international": "case-by-case, no visa sponsorship",
        },
        "experience": {
            "range_years": (5, 9),
            "ideal_total": (6, 8),
            "ideal_ml_years": (4, 5),
            "requires_product_company": True,
        },
        "must_have_skills": [
            # Embeddings & retrieval
            "embeddings", "sentence-transformers", "BGE", "E5",
            "semantic search", "vector search", "retrieval",
            # Vector databases
            "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
            "Elasticsearch", "FAISS", "vector database",
            # Core
            "Python", "NLP", "information retrieval",
            # Evaluation
            "NDCG", "MRR", "MAP", "ranking evaluation", "A/B testing",
            # Systems
            "ranking system", "recommendation system", "search system",
        ],
        "nice_to_have_skills": [
            "LoRA", "QLoRA", "PEFT", "LLM fine-tuning",
            "XGBoost", "learning-to-rank",
            "HR-tech", "recruiting", "marketplace",
            "distributed systems", "inference optimization",
            "open-source contributions",
            "RAG", "retrieval augmented generation",
            "hybrid search", "BM25",
        ],
        "relevant_titles": [
            "AI Engineer", "ML Engineer", "Machine Learning Engineer",
            "Senior AI Engineer", "Senior ML Engineer",
            "Data Scientist", "Applied Scientist",
            "NLP Engineer", "Search Engineer", "Ranking Engineer",
            "Research Engineer", "ML Researcher",
            "Software Engineer",  # if in ML/AI context
            "Backend Engineer",  # if in ML/AI context
            "Data Engineer",  # adjacent
        ],
        "disqualifier_companies": [
            "TCS", "Tata Consultancy Services",
            "Infosys",
            "Wipro",
            "Accenture",
            "Cognizant",
            "Capgemini",
            "HCL", "HCL Technologies",
            "Tech Mahindra",
        ],
        "disqualifier_domains": [
            "computer vision only",
            "speech only",
            "robotics only",
        ],
        "notice_period": {
            "ideal_max_days": 30,
            "acceptable_max_days": 60,
            "penalty_above_days": 90,
        },
    }
