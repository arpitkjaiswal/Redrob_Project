"""
Task 2: Data Cleaning & Preprocessing

Normalizes, standardizes, and validates candidate data.
Detects anomalies and prepares clean data for feature extraction.
"""

import re
import logging
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Skill name normalization map ────────────────────────────────────────────
# Maps common variations to canonical names
SKILL_NORMALIZATION = {
    # Python ecosystem
    "python3": "Python", "python 3": "Python", "py": "Python",
    "pytorch": "PyTorch", "py torch": "PyTorch",
    "tensorflow": "TensorFlow", "tf": "TensorFlow", "tensor flow": "TensorFlow",
    "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn", "scikit learn": "Scikit-learn",
    "numpy": "NumPy", "pandas": "Pandas",
    # ML/AI
    "ml": "Machine Learning", "machine learning": "Machine Learning",
    "dl": "Deep Learning", "deep learning": "Deep Learning",
    "nlp": "NLP", "natural language processing": "NLP",
    "llm": "LLM", "large language model": "LLM", "large language models": "LLM",
    "genai": "Generative AI", "gen ai": "Generative AI", "generative ai": "Generative AI",
    "rag": "RAG", "retrieval augmented generation": "RAG",
    "cv": "Computer Vision", "computer vision": "Computer Vision",
    "gans": "GANs", "gan": "GANs", "generative adversarial": "GANs",
    "cnn": "CNN", "convolutional neural network": "CNN",
    "rnn": "RNN", "recurrent neural network": "RNN",
    "transformer": "Transformers", "transformers": "Transformers",
    "bert": "BERT", "gpt": "GPT",
    "lora": "LoRA", "qlora": "QLoRA", "peft": "PEFT",
    "fine-tuning": "Fine-tuning LLMs", "finetuning": "Fine-tuning LLMs",
    "fine tuning": "Fine-tuning LLMs",
    # Embeddings & retrieval
    "sentence-transformers": "Sentence-Transformers",
    "sentence transformers": "Sentence-Transformers",
    "faiss": "FAISS", "pinecone": "Pinecone",
    "weaviate": "Weaviate", "qdrant": "Qdrant",
    "milvus": "Milvus", "opensearch": "OpenSearch",
    "elasticsearch": "Elasticsearch", "elastic search": "Elasticsearch",
    "vector db": "Vector Database", "vector database": "Vector Database",
    "chromadb": "ChromaDB", "chroma": "ChromaDB",
    # Data engineering
    "spark": "Apache Spark", "pyspark": "PySpark", "apache spark": "Apache Spark",
    "airflow": "Airflow", "apache airflow": "Airflow",
    "kafka": "Kafka", "apache kafka": "Kafka",
    "hadoop": "Hadoop", "apache hadoop": "Hadoop",
    # Cloud
    "aws": "AWS", "amazon web services": "AWS",
    "gcp": "GCP", "google cloud": "GCP", "google cloud platform": "GCP",
    "azure": "Azure", "microsoft azure": "Azure",
    # Databases
    "sql": "SQL", "mysql": "MySQL", "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL", "mongodb": "MongoDB", "mongo": "MongoDB",
    "redis": "Redis", "snowflake": "Snowflake",
    # Web
    "react": "React", "reactjs": "React", "react.js": "React",
    "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
    "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "angular": "Angular", "angularjs": "Angular",
    "vue": "Vue.js", "vuejs": "Vue.js",
    # DevOps
    "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "ci/cd": "CI/CD", "cicd": "CI/CD",
    "git": "Git", "github": "GitHub",
    # Evaluation & ranking
    "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "a/b testing": "A/B Testing", "ab testing": "A/B Testing",
}


def normalize_text(text: str) -> str:
    """Normalize text: strip, collapse whitespace, basic cleanup."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_skill_name(skill_name: str) -> str:
    """Normalize a skill name to its canonical form."""
    if not skill_name:
        return ""
    clean = skill_name.strip()
    lookup = clean.lower()
    return SKILL_NORMALIZATION.get(lookup, clean)


def parse_date_safe(date_str: Optional[str]) -> Optional[date]:
    """Parse a date string safely, returning None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        try:
            return datetime.strptime(date_str, "%Y/%m/%d").date()
        except (ValueError, TypeError):
            return None


def compute_months_between(start_str: str, end_str: Optional[str]) -> Optional[int]:
    """Compute months between two date strings."""
    start = parse_date_safe(start_str)
    if not start:
        return None
    end = parse_date_safe(end_str) if end_str else date.today()
    if not end:
        return None
    delta = (end.year - start.year) * 12 + (end.month - start.month)
    return max(0, delta)


def clean_candidate_dict(data: dict) -> dict:
    """
    Clean and normalize a raw candidate dict in-place.
    Returns the cleaned dict.
    """
    # --- Profile ---
    profile = data.get("profile", {})
    for field in ["headline", "summary", "location", "country",
                  "current_title", "current_company", "current_industry"]:
        if field in profile:
            profile[field] = normalize_text(str(profile.get(field, "")))

    # Ensure years_of_experience is a float
    yoe = profile.get("years_of_experience", 0)
    try:
        profile["years_of_experience"] = float(yoe)
    except (ValueError, TypeError):
        profile["years_of_experience"] = 0.0

    # --- Career History ---
    for entry in data.get("career_history", []):
        for field in ["company", "title", "industry", "description"]:
            if field in entry:
                entry[field] = normalize_text(str(entry.get(field, "")))

        # Validate duration_months against dates
        computed = compute_months_between(
            entry.get("start_date"), entry.get("end_date")
        )
        if computed is not None:
            stated = entry.get("duration_months", 0)
            # Flag large discrepancies (>12 months off)
            if abs(stated - computed) > 12:
                entry["_duration_mismatch"] = True
                entry["_computed_duration"] = computed

    # --- Skills ---
    for skill in data.get("skills", []):
        skill["name"] = normalize_skill_name(skill.get("name", ""))
        # Ensure numeric fields
        try:
            skill["endorsements"] = max(0, int(skill.get("endorsements", 0)))
        except (ValueError, TypeError):
            skill["endorsements"] = 0
        try:
            skill["duration_months"] = max(0, int(skill.get("duration_months", 0)))
        except (ValueError, TypeError):
            skill["duration_months"] = 0

    # --- Education ---
    for edu in data.get("education", []):
        for field in ["institution", "degree", "field_of_study"]:
            if field in edu:
                edu[field] = normalize_text(str(edu.get(field, "")))

    return data


def detect_anomalies(data: dict) -> list[str]:
    """
    Detect data anomalies that might indicate honeypots or data quality issues.
    Returns a list of anomaly descriptions.
    """
    anomalies = []
    profile = data.get("profile", {})
    career = data.get("career_history", [])
    skills = data.get("skills", [])
    signals = data.get("redrob_signals", {})

    # 1. Experience vs career history mismatch
    stated_yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(e.get("duration_months", 0) for e in career)
    career_yoe = total_career_months / 12.0
    if stated_yoe > 0 and career_yoe > 0:
        if abs(stated_yoe - career_yoe) > 5:
            anomalies.append(
                f"YOE mismatch: stated={stated_yoe:.1f}yr, career_sum={career_yoe:.1f}yr"
            )

    # 2. Expert skills with 0 duration
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_duration >= 3:
        anomalies.append(
            f"{expert_zero_duration} expert skills with 0 months duration"
        )

    # 3. Too many expert skills (>8 is very suspicious)
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count > 8:
        anomalies.append(f"{expert_count} expert-level skills (suspiciously high)")

    # 4. Career date impossibilities
    for entry in career:
        if entry.get("_duration_mismatch"):
            anomalies.append(
                f"Duration mismatch at {entry.get('company', '?')}: "
                f"stated={entry.get('duration_months')}mo, "
                f"computed={entry.get('_computed_duration')}mo"
            )

    # 5. Title-description mismatch (major red flag)
    for entry in career:
        title_lower = entry.get("title", "").lower()
        desc_lower = entry.get("description", "").lower()
        # Marketing Manager with ML descriptions
        non_tech_titles = ["marketing manager", "hr manager", "accountant",
                          "sales executive", "content writer", "graphic designer",
                          "operations manager", "customer support", "civil engineer",
                          "mechanical engineer"]
        ml_keywords = ["machine learning", "deep learning", "neural network",
                       "model training", "embeddings", "nlp", "computer vision",
                       "data pipeline", "ml model"]

        is_non_tech = any(t in title_lower for t in non_tech_titles)
        has_ml_desc = sum(1 for kw in ml_keywords if kw in desc_lower) >= 2

        if is_non_tech and has_ml_desc:
            anomalies.append(
                f"Title-description mismatch: title='{entry.get('title')}' "
                f"but description mentions ML concepts"
            )

    # 6. Skill assessment vs proficiency mismatch
    assessments = signals.get("skill_assessment_scores", {})
    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        if name in assessments:
            score = assessments[name]
            if prof == "expert" and score < 30:
                anomalies.append(
                    f"Skill '{name}': claims expert but assessment score={score}"
                )
            elif prof == "advanced" and score < 20:
                anomalies.append(
                    f"Skill '{name}': claims advanced but assessment score={score}"
                )

    return anomalies
