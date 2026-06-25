"""
src/skill_matcher.py
Semantic skill clusters and trust-weighted skill scoring.
Maps any skill string to a canonical cluster, then scores it
based on proficiency + endorsements + duration.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

# ─── Skill clusters ────────────────────────────────────────────────────────
# Each cluster → list of keywords/phrases (all lower-case).
# A skill is mapped to the FIRST cluster whose any keyword is a substring match.

SKILL_CLUSTERS: dict[str, list[str]] = {
    "llm_frameworks": [
        "langchain", "llamaindex", "llama_index", "haystack",
        "semantic kernel", "autogen", "crewai", "dspy",
    ],
    "vector_databases": [
        "faiss", "pinecone", "weaviate", "qdrant", "chroma",
        "milvus", "pgvector", "vespa", "elasticsearch vector",
    ],
    "foundation_models": [
        "gpt-4", "gpt-3", "openai", "claude", "gemini", "palm",
        "llama", "mistral", "falcon", "bloom", "bert", "roberta",
        "t5", "bart", "whisper", "stable diffusion", "dall-e",
        "hugging face", "huggingface", "transformers",
    ],
    "mlops": [
        "mlflow", "kubeflow", "sagemaker", "vertex ai", "wandb",
        "dvc", "bentoml", "seldon", "triton", "torchserve",
        "ray serve", "clearml",
    ],
    "deep_learning": [
        "pytorch", "tensorflow", "keras", "jax", "flax",
        "caffe", "mxnet", "onnx",
    ],
    "classical_ml": [
        "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
        "random forest", "gradient boosting", "svm", "logistic regression",
    ],
    "data_engineering": [
        "spark", "kafka", "airflow", "dbt", "flink", "hadoop",
        "databricks", "snowflake", "bigquery", "redshift", "dask",
        "pandas", "polars",
    ],
    "cloud_platforms": [
        "aws", "azure", "gcp", "google cloud", "amazon web services",
        "ec2", "s3", "lambda", "ecs", "eks", "gke", "aks",
    ],
    "mlops_infra": [
        "docker", "kubernetes", "k8s", "helm", "terraform",
        "ci/cd", "github actions", "jenkins", "argo",
    ],
    "languages": [
        "python", "rust", "go", "golang", "scala", "java", "c++",
        "c#", "sql", "r ", " r,", "julia",
    ],
    "production_systems": [
        "rag", "retrieval augmented", "fine-tuning", "finetuning",
        "lora", "rlhf", "instruction tuning", "prompt engineering",
        "a/b testing", "feature store", "online learning",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "cassandra",
        "dynamodb", "neo4j", "sqlite",
    ],
}

PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.55,
    "beginner": 0.25,
}

# Minimum endorsements to trust a proficiency claim
ENDORSEMENT_THRESHOLD = 3

# Duration at which score fully saturates (months)
DURATION_SATURATION = 36


@lru_cache(maxsize=4096)
def map_skill_to_cluster(skill_name: str) -> Optional[str]:
    """Return the cluster name for a skill string, or None if unrecognized."""
    norm = skill_name.lower().strip()
    for cluster, keywords in SKILL_CLUSTERS.items():
        for kw in keywords:
            if kw in norm or norm in kw:
                return cluster
    return None


def _duration_factor(duration_months: int) -> float:
    """Sigmoid-like saturation: approaches 1.0 at DURATION_SATURATION months."""
    if duration_months <= 0:
        return 0.05
    ratio = min(duration_months / DURATION_SATURATION, 1.0)
    return 0.05 + 0.95 * ratio


def _endorsement_factor(endorsements: int, proficiency: str) -> float:
    """
    Reduce weight if someone claims 'expert'/'advanced' without endorsements.
    """
    base = PROFICIENCY_WEIGHT.get(proficiency, 0.3)
    if proficiency in ("expert", "advanced") and endorsements < ENDORSEMENT_THRESHOLD:
        # Penalty: cap at 'intermediate' trust level
        base = min(base, PROFICIENCY_WEIGHT["intermediate"])
    elif endorsements >= 10:
        base = min(base * 1.1, 1.0)   # slight boost for highly endorsed skills
    return base


def score_skill(skill: dict) -> tuple[float, Optional[str]]:
    """
    Return (trust_weighted_score, cluster_name) for a single skill dict.
    Score range: [0, 1].
    """
    name = skill.get("name", "")
    proficiency = skill.get("proficiency", "beginner")
    endorsements = int(skill.get("endorsements", 0) or 0)
    duration_months = int(skill.get("duration_months", 0) or 0)

    cluster = map_skill_to_cluster(name)

    prof_factor = _endorsement_factor(endorsements, proficiency)
    dur_factor = _duration_factor(duration_months)

    # Geometric mean keeps both factors important
    score = (prof_factor * dur_factor) ** 0.5
    return round(score, 4), cluster


def compute_skill_score(skills: list[dict]) -> dict:
    """
    Compute an aggregated skill signal for a candidate.

    Returns:
        {
          'total_score': float,        # [0, 1] normalized
          'cluster_coverage': int,     # number of distinct clusters hit
          'cluster_scores': dict,      # cluster → best score in that cluster
          'top_skills': list[str],     # names of top-5 highest scoring skills
        }
    """
    cluster_scores: dict[str, float] = {}
    skill_scores: list[tuple[float, str]] = []   # (score, name)

    for sk in (skills or []):
        s, cluster = score_skill(sk)
        skill_scores.append((s, sk.get("name", "")))
        if cluster:
            cluster_scores[cluster] = max(cluster_scores.get(cluster, 0.0), s)

    if not skill_scores:
        return {
            "total_score": 0.0,
            "cluster_coverage": 0,
            "cluster_scores": {},
            "top_skills": [],
        }

    # Aggregate: average of per-cluster bests, boosted by coverage breadth
    if cluster_scores:
        avg_cluster = sum(cluster_scores.values()) / len(cluster_scores)
        coverage_boost = min(len(cluster_scores) / len(SKILL_CLUSTERS), 1.0) * 0.2
        total_score = min(avg_cluster + coverage_boost, 1.0)
    else:
        # No recognized clusters — plain average of raw skill scores
        total_score = sum(s for s, _ in skill_scores) / len(skill_scores) * 0.4

    # Top skills by score
    skill_scores.sort(reverse=True)
    top_skills = [name for _, name in skill_scores[:5]]

    return {
        "total_score": round(total_score, 4),
        "cluster_coverage": len(cluster_scores),
        "cluster_scores": cluster_scores,
        "top_skills": top_skills,
    }
