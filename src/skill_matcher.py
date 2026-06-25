"""
Task 6: Skill-Matching Approaches

Multi-level skill matching that goes beyond keyword comparison:
1. Exact match against JD requirements
2. Semantic synonym matching via curated dictionary
3. Contextual skill inference from career descriptions
4. Trust-weighted scoring (endorsements, duration, assessments)
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Semantic Skill Clusters ────────────────────────────────────────────────
# Maps JD requirement concepts to clusters of related skill names/keywords.
# A candidate matching ANY skill in a cluster gets credit for that concept.

SKILL_CLUSTERS = {
    # --- Must-have clusters (from JD) ---
    "embeddings_retrieval": {
        "skills": [
            "embeddings", "sentence-transformers", "Sentence-Transformers",
            "BGE", "E5", "semantic search", "vector search",
            "retrieval", "information retrieval", "dense retrieval",
            "embedding", "text embeddings", "OpenAI embeddings",
            "word2vec", "GloVe", "doc2vec",
        ],
        "career_keywords": [
            "embedding", "retrieval", "semantic search", "vector search",
            "dense retrieval", "similarity search", "nearest neighbor",
            "sentence transformer", "text representation",
        ],
        "weight": 1.0,  # Must-have
    },
    "vector_databases": {
        "skills": [
            "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
            "Elasticsearch", "FAISS", "Vector Database", "ChromaDB",
            "Annoy", "ScaNN", "vector store", "vector index",
        ],
        "career_keywords": [
            "vector database", "vector store", "faiss", "pinecone",
            "weaviate", "qdrant", "milvus", "elasticsearch",
            "opensearch", "similarity index", "ann index",
            "approximate nearest neighbor",
        ],
        "weight": 1.0,
    },
    "python_strong": {
        "skills": [
            "Python", "PySpark", "Flask", "Django", "FastAPI",
        ],
        "career_keywords": [
            "python", "flask", "django", "fastapi", "pyspark",
        ],
        "weight": 0.8,
    },
    "nlp_ir": {
        "skills": [
            "NLP", "Natural Language Processing", "information retrieval",
            "text mining", "text classification", "named entity recognition",
            "sentiment analysis", "topic modeling", "text processing",
            "BERT", "GPT", "Transformers", "Hugging Face",
            "spaCy", "NLTK",
        ],
        "career_keywords": [
            "nlp", "natural language", "text classification",
            "named entity", "sentiment", "language model",
            "text mining", "information retrieval", "search relevance",
            "query understanding", "document ranking",
        ],
        "weight": 1.0,
    },
    "ranking_evaluation": {
        "skills": [
            "NDCG", "MRR", "MAP", "ranking evaluation",
            "A/B Testing", "search relevance",
            "precision", "recall", "evaluation metrics",
        ],
        "career_keywords": [
            "ndcg", "mrr", "mean average precision", "ranking evaluation",
            "a/b test", "search quality", "relevance metric",
            "offline evaluation", "online evaluation",
            "evaluation framework", "benchmark",
        ],
        "weight": 0.9,
    },
    "ranking_recommendation_systems": {
        "skills": [
            "recommendation system", "ranking system", "search system",
            "collaborative filtering", "content-based filtering",
            "learning-to-rank", "XGBoost", "LightGBM",
        ],
        "career_keywords": [
            "recommendation", "ranking system", "search system",
            "ranker", "re-ranking", "learning to rank",
            "candidate retrieval", "matching system",
            "recommender", "personalization",
        ],
        "weight": 1.0,
    },

    # --- Nice-to-have clusters ---
    "llm_finetuning": {
        "skills": [
            "LoRA", "QLoRA", "PEFT", "Fine-tuning LLMs", "LLM",
            "Generative AI", "RAG", "prompt engineering",
            "Langchain", "LlamaIndex",
        ],
        "career_keywords": [
            "fine-tuning", "fine tuning", "finetuning", "lora", "qlora",
            "peft", "llm", "large language model", "prompt engineering",
            "rag", "retrieval augmented", "langchain",
        ],
        "weight": 0.6,
    },
    "ml_production": {
        "skills": [
            "MLOps", "ML Engineering", "Model Deployment",
            "Docker", "Kubernetes", "CI/CD",
            "BentoML", "MLflow", "Weights & Biases",
            "TensorFlow Serving", "TorchServe",
        ],
        "career_keywords": [
            "production", "deploy", "serving", "inference",
            "mlops", "pipeline", "model serving",
            "containeriz", "docker", "kubernetes",
        ],
        "weight": 0.5,
    },
    "data_engineering": {
        "skills": [
            "Apache Spark", "PySpark", "Airflow", "Kafka",
            "Hadoop", "Snowflake", "dbt", "ETL",
            "Data Pipeline", "Data Engineering",
        ],
        "career_keywords": [
            "data pipeline", "etl", "data engineering",
            "spark", "airflow", "kafka", "data warehouse",
            "batch processing", "stream processing",
        ],
        "weight": 0.4,
    },
    "deep_learning_core": {
        "skills": [
            "Deep Learning", "PyTorch", "TensorFlow",
            "Neural Networks", "CNN", "RNN", "Transformers",
            "Machine Learning", "Scikit-learn",
            "Statistical Modeling", "Feature Engineering",
        ],
        "career_keywords": [
            "deep learning", "neural network", "pytorch", "tensorflow",
            "model training", "feature engineering",
            "machine learning", "classification", "regression",
        ],
        "weight": 0.7,
    },
}

# Proficiency score mapping
PROFICIENCY_SCORES = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.5,
    "beginner": 0.25,
}


def match_skills_to_clusters(
    candidate_skills: list[dict],
    career_descriptions: list[str],
) -> dict:
    """
    Match a candidate's skills and career descriptions against all skill clusters.

    Returns dict with:
      - cluster_name -> {
            "matched": bool,
            "skill_matches": [...],
            "career_matches": [...],
            "score": float (0-1),
            "weight": float,
        }
    """
    # Build lowercase skill lookup
    skill_lookup = {}
    for s in candidate_skills:
        name = s.get("name", "").strip()
        if name:
            skill_lookup[name.lower()] = s

    # Build combined career text
    career_text = " ".join(career_descriptions).lower()

    results = {}

    for cluster_name, cluster_def in SKILL_CLUSTERS.items():
        skill_matches = []
        career_matches = []

        # Check explicit skill matches
        for cluster_skill in cluster_def["skills"]:
            key = cluster_skill.lower()
            if key in skill_lookup:
                s = skill_lookup[key]
                skill_matches.append({
                    "name": s.get("name", ""),
                    "proficiency": s.get("proficiency", "beginner"),
                    "endorsements": s.get("endorsements", 0),
                    "duration_months": s.get("duration_months", 0),
                })

        # Check career description for contextual matches
        for keyword in cluster_def.get("career_keywords", []):
            if keyword.lower() in career_text:
                career_matches.append(keyword)

        # Compute cluster score
        has_skill_match = len(skill_matches) > 0
        has_career_match = len(career_matches) > 0
        matched = has_skill_match or has_career_match

        if matched:
            # Skill match score: weighted by proficiency, endorsements, duration
            skill_score = 0.0
            if skill_matches:
                best_skill = max(
                    skill_matches,
                    key=lambda s: (
                        PROFICIENCY_SCORES.get(s["proficiency"], 0.25),
                        min(s["endorsements"], 50) / 50.0,
                        min(s["duration_months"], 60) / 60.0,
                    ),
                )
                prof_score = PROFICIENCY_SCORES.get(best_skill["proficiency"], 0.25)
                endorse_score = min(best_skill["endorsements"], 50) / 50.0
                duration_score = min(best_skill["duration_months"], 60) / 60.0
                skill_score = (
                    0.5 * prof_score
                    + 0.25 * endorse_score
                    + 0.25 * duration_score
                )
                # Bonus for multiple skill matches in cluster
                skill_score = min(1.0, skill_score + 0.1 * (len(skill_matches) - 1))

            # Career match score
            career_score = min(1.0, len(career_matches) * 0.2) if has_career_match else 0.0

            # Combined score (explicit skills weighted higher than career inference)
            if has_skill_match and has_career_match:
                score = 0.7 * skill_score + 0.3 * career_score
            elif has_skill_match:
                score = 0.8 * skill_score  # Slight penalty for no career evidence
            else:
                score = 0.5 * career_score  # Career-only is weaker signal
        else:
            score = 0.0

        results[cluster_name] = {
            "matched": matched,
            "skill_matches": skill_matches,
            "career_matches": list(set(career_matches)),
            "score": round(score, 4),
            "weight": cluster_def["weight"],
        }

    return results


def compute_skill_trust_score(skill: dict, assessment_scores: dict) -> float:
    """
    Compute a trust-weighted score for a single skill.
    Catches keyword stuffers who list skills they don't actually have.
    """
    name = skill.get("name", "")
    proficiency = skill.get("proficiency", "beginner")
    endorsements = skill.get("endorsements", 0)
    duration = skill.get("duration_months", 0)

    prof_score = PROFICIENCY_SCORES.get(proficiency, 0.25)

    # Red flags: high proficiency with low evidence
    trust_penalty = 1.0

    # Expert/advanced with 0 endorsements is suspicious
    if proficiency in ("expert", "advanced") and endorsements == 0:
        trust_penalty *= 0.5

    # Expert with very low duration is suspicious
    if proficiency == "expert" and duration < 6:
        trust_penalty *= 0.4

    # Advanced with very low duration
    if proficiency == "advanced" and duration < 3:
        trust_penalty *= 0.6

    # Assessment score mismatch
    if name in assessment_scores:
        assessment = assessment_scores[name]
        if proficiency == "expert" and assessment < 40:
            trust_penalty *= 0.3
        elif proficiency == "advanced" and assessment < 30:
            trust_penalty *= 0.4
        elif proficiency == "intermediate" and assessment < 20:
            trust_penalty *= 0.6

    # Positive signal: high endorsements + high duration + good assessment
    trust_bonus = 1.0
    if endorsements > 20 and duration > 24:
        trust_bonus = 1.2
    if name in assessment_scores and assessment_scores[name] > 70:
        trust_bonus *= 1.1

    return round(prof_score * trust_penalty * min(trust_bonus, 1.3), 4)


def detect_keyword_stuffer(
    candidate_skills: list[dict],
    career_history: list[dict],
    current_title: str,
) -> float:
    """
    Detect keyword-stuffer candidates: people who list many AI/ML skills
    but whose actual career and title don't support them.

    Returns a stuffer_score from 0.0 (not a stuffer) to 1.0 (definite stuffer).
    """
    # Count AI-related skills
    ai_skill_names = set()
    for cluster in SKILL_CLUSTERS.values():
        for s in cluster["skills"]:
            ai_skill_names.add(s.lower())

    candidate_ai_skills = [
        s for s in candidate_skills
        if s.get("name", "").lower() in ai_skill_names
    ]
    ai_skill_count = len(candidate_ai_skills)

    if ai_skill_count < 3:
        return 0.0  # Not enough AI skills to be a stuffer

    # Check if title is non-technical
    non_tech_titles = [
        "marketing manager", "hr manager", "accountant",
        "sales executive", "content writer", "graphic designer",
        "operations manager", "customer support", "civil engineer",
        "mechanical engineer", "business analyst",
    ]
    title_lower = current_title.lower()
    is_non_tech_title = any(t in title_lower for t in non_tech_titles)

    # Check career descriptions for actual AI work
    career_text = " ".join(e.get("description", "") for e in career_history).lower()
    ai_career_keywords = [
        "machine learning", "deep learning", "neural network",
        "model training", "model deployment", "embeddings",
        "nlp", "natural language", "recommendation system",
        "ranking system", "search system", "data science",
        "ml pipeline", "ml model", "ai system",
    ]
    career_ai_mentions = sum(1 for kw in ai_career_keywords if kw in career_text)

    # Stuffer heuristic
    stuffer_score = 0.0

    if is_non_tech_title and ai_skill_count >= 5:
        stuffer_score += 0.4

    if is_non_tech_title and career_ai_mentions < 2:
        stuffer_score += 0.3

    # Many AI skills but no career evidence
    if ai_skill_count >= 6 and career_ai_mentions < 2:
        stuffer_score += 0.3

    # Expert in many AI skills (unlikely for non-tech roles)
    expert_ai = sum(
        1 for s in candidate_ai_skills if s.get("proficiency") == "expert"
    )
    if expert_ai >= 3 and is_non_tech_title:
        stuffer_score += 0.2

    return min(1.0, stuffer_score)
