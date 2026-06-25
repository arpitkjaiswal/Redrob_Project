"""
src/feature_extractor.py
Extract 50+ numerical/boolean features from a cleaned candidate record.
All features normalized to [0, 1] where possible.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

from src.data_cleaner import normalize_text, parse_date, date_diff_months
from src.skill_matcher import compute_skill_score

# ─── Production signal keywords ─────────────────────────────────────────────
PRODUCTION_KEYWORDS = [
    "deployed", "production", "serving", "latency", "throughput",
    "scale", "million", "billion", "real-time", "realtime",
    "inference", "endpoint", "api", "microservice", "pipeline",
    "streaming", "kafka", "monitoring", "alerting", "sla",
    "a/b test", "rollout", "canary", "blue-green",
]

ML_KEYWORDS = [
    "model", "train", "fine-tun", "rag", "embedding", "llm",
    "neural", "deep learning", "machine learning", "nlp", "cv",
    "computer vision", "reinforcement", "gpt", "bert", "transformer",
    "feature engineering", "experimentation",
]

PRODUCT_COMPANY_SIGNALS = [
    "netflix", "google", "meta", "facebook", "amazon", "apple",
    "microsoft", "uber", "airbnb", "stripe", "openai", "anthropic",
    "flipkart", "swiggy", "zomato", "meesho", "razorpay",
    "phonepe", "paytm", "ola", "byju", "zepto", "cred",
]

TIER1_INSTITUTIONS = {
    "iit", "iim", "bits pilani", "nit", "iisc", "stanford",
    "mit", "carnegie mellon", "cmu", "berkeley", "oxford",
    "cambridge", "georgia tech", "ethz", "caltech",
}

TIER2_INSTITUTIONS = {
    "vit", "manipal", "srm", "amity", "symbiosis",
    "university of", "institute of technology",
}

BIG_COMPANY_SIZES = {"1001-5000", "5001-10000", "10001+"}
CONSULTING_TITLES = {"consultant", "analyst", "associate", "manager"}


def _kw_density(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found in text (normalized to [0,1])."""
    if not text or not keywords:
        return 0.0
    text_norm = normalize_text(text)
    hits = sum(1 for kw in keywords if kw in text_norm)
    return hits / len(keywords)


def _log_norm(value: float, max_val: float) -> float:
    """Log-normalize a value to approximately [0, 1]."""
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(max_val), 1.0)


def extract_career_features(career: list[dict], profile: dict) -> dict[str, Any]:
    """Features derived from career history."""
    today = date.today()
    f: dict[str, Any] = {}

    if not career:
        return {
            "total_career_months": 0,
            "num_jobs": 0,
            "avg_tenure_months": 0,
            "longest_tenure_months": 0,
            "has_current_job": False,
            "production_signal_score": 0.0,
            "ml_signal_score": 0.0,
            "product_company_score": 0.0,
            "big_company_ratio": 0.0,
            "job_hopping_penalty": 0.0,
            "is_consulting_heavy": False,
            "career_score": 0.0,
        }

    # Collect all descriptions
    all_descs = " ".join(j.get("description", "") for j in career)

    # Production & ML signals from career text
    f["production_signal_score"] = min(_kw_density(all_descs, PRODUCTION_KEYWORDS) * 5, 1.0)
    f["ml_signal_score"] = min(_kw_density(all_descs, ML_KEYWORDS) * 4, 1.0)

    # Product company detection
    all_companies = " ".join(
        normalize_text(j.get("company", "")) for j in career
    )
    pc_hits = sum(1 for sig in PRODUCT_COMPANY_SIGNALS if sig in all_companies)
    f["product_company_score"] = min(pc_hits / 3, 1.0)

    # Big company ratio
    big_count = sum(
        1 for j in career if j.get("company_size", "") in BIG_COMPANY_SIZES
    )
    f["big_company_ratio"] = big_count / len(career)

    # Tenure
    tenures = []
    for j in career:
        start = parse_date(j.get("start_date"))
        end_raw = j.get("end_date")
        end = today if (j.get("is_current") or not end_raw) else parse_date(end_raw)
        months = date_diff_months(start, end)
        tenures.append(months)

    f["num_jobs"] = len(career)
    f["total_career_months"] = sum(tenures)
    f["avg_tenure_months"] = sum(tenures) / len(tenures) if tenures else 0
    f["longest_tenure_months"] = max(tenures) if tenures else 0
    f["has_current_job"] = any(j.get("is_current") for j in career)

    # Job hopping penalty: penalize avg tenure < 12 months
    avg = f["avg_tenure_months"]
    f["job_hopping_penalty"] = max(0.0, (12 - avg) / 12) if avg < 12 else 0.0

    # Consulting-heavy detection
    all_titles = " ".join(normalize_text(j.get("title", "")) for j in career)
    consulting_hits = sum(1 for t in CONSULTING_TITLES if t in all_titles)
    f["is_consulting_heavy"] = consulting_hits >= 2 and f["product_company_score"] < 0.2

    # Composite career score
    career_score = (
        f["production_signal_score"] * 0.30
        + f["ml_signal_score"] * 0.25
        + f["product_company_score"] * 0.20
        + f["big_company_ratio"] * 0.10
        + min(f["total_career_months"] / (10 * 12), 1.0) * 0.15
        - f["job_hopping_penalty"] * 0.15
        - (0.2 if f["is_consulting_heavy"] else 0)
    )
    f["career_score"] = max(0.0, min(career_score, 1.0))

    return f


def extract_education_features(education: list[dict]) -> dict[str, Any]:
    """Features derived from education."""
    f: dict[str, Any] = {}

    if not education:
        return {
            "highest_degree_score": 0.0,
            "institution_tier_score": 0.0,
            "relevant_field": False,
            "education_score": 0.0,
        }

    DEGREE_SCORES = {
        "phd": 1.0, "ph.d": 1.0, "doctor": 1.0,
        "master": 0.75, "m.tech": 0.75, "m.s": 0.75, "msc": 0.75,
        "bachelor": 0.5, "b.tech": 0.5, "b.e": 0.5, "b.sc": 0.5,
        "diploma": 0.2,
    }

    best_degree = 0.0
    best_tier = 0.0
    relevant = False

    RELEVANT_FIELDS = {
        "computer", "software", "machine learning", "data science",
        "artificial intelligence", "statistics", "mathematics", "physics",
        "electrical", "electronics",
    }

    TIER_SCORES = {"tier_1": 1.0, "tier_2": 0.65, "tier_3": 0.35,
                   "tier_4": 0.15, "unknown": 0.1}

    for edu in education:
        degree_norm = normalize_text(edu.get("degree", ""))
        for key, score in DEGREE_SCORES.items():
            if key in degree_norm:
                best_degree = max(best_degree, score)
                break

        # Institution tier (from schema field OR keyword match)
        tier = edu.get("tier", "unknown") or "unknown"
        tier_score = TIER_SCORES.get(tier, 0.1)

        # Also check institution name directly
        inst_norm = normalize_text(edu.get("institution", ""))
        for t1 in TIER1_INSTITUTIONS:
            if t1 in inst_norm:
                tier_score = max(tier_score, 1.0)
                break
        for t2 in TIER2_INSTITUTIONS:
            if t2 in inst_norm:
                tier_score = max(tier_score, 0.65)
                break

        best_tier = max(best_tier, tier_score)

        # Relevant field
        field_norm = normalize_text(edu.get("field_of_study", ""))
        if any(rf in field_norm for rf in RELEVANT_FIELDS):
            relevant = True

    f["highest_degree_score"] = best_degree
    f["institution_tier_score"] = best_tier
    f["relevant_field"] = relevant

    edu_score = (
        best_degree * 0.45
        + best_tier * 0.35
        + (0.2 if relevant else 0.0)
    )
    f["education_score"] = round(min(edu_score, 1.0), 4)
    return f


def extract_behavioral_features(signals: dict) -> dict[str, Any]:
    """Features from redrob_signals (platform behavioral data)."""
    f: dict[str, Any] = {}

    f["open_to_work"] = bool(signals.get("open_to_work_flag", False))
    f["willing_to_relocate"] = bool(signals.get("willing_to_relocate", False))

    # Profile completeness [0, 1]
    f["profile_completeness"] = float(
        signals.get("profile_completeness_score", 0) or 0
    ) / 100.0

    # Engagement metrics (log-normalized)
    f["profile_views_norm"] = _log_norm(
        float(signals.get("profile_views_received_30d", 0) or 0), 500
    )
    f["saved_by_recruiters_norm"] = _log_norm(
        float(signals.get("saved_by_recruiters_30d", 0) or 0), 50
    )
    f["search_appearance_norm"] = _log_norm(
        float(signals.get("search_appearance_30d", 0) or 0), 1000
    )

    # Responsiveness
    recruiter_rr = float(signals.get("recruiter_response_rate", 0) or 0)
    avg_response_h = float(signals.get("avg_response_time_hours", 999) or 999)
    f["recruiter_responsiveness"] = (
        recruiter_rr * 0.6 + max(0, 1 - avg_response_h / 72) * 0.4
    )

    # Interview & offer reliability
    f["interview_completion_rate"] = float(
        signals.get("interview_completion_rate", 0) or 0
    )
    offer_acc = float(signals.get("offer_acceptance_rate", -1) or -1)
    f["offer_acceptance_rate"] = max(offer_acc, 0.0)  # -1 → 0

    # GitHub activity [0, 1]
    github = float(signals.get("github_activity_score", -1) or -1)
    f["github_activity"] = max(github, 0.0) / 100.0

    # Trust signals
    f["verified"] = int(
        bool(signals.get("verified_email")) and bool(signals.get("verified_phone"))
    )
    f["linkedin_connected"] = int(bool(signals.get("linkedin_connected")))

    # Notice period penalty: > 90 days is a soft negative
    notice = int(signals.get("notice_period_days", 0) or 0)
    f["notice_period_days"] = notice
    f["notice_penalty"] = min(max(notice - 90, 0) / 90, 1.0)

    # Skill assessment scores (average of all completed assessments)
    assessments = signals.get("skill_assessment_scores", {}) or {}
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments) / 100.0
    else:
        avg_assessment = 0.0
    f["avg_skill_assessment"] = avg_assessment

    # Composite behavioral score
    b = (
        f["profile_completeness"] * 0.10
        + f["recruiter_responsiveness"] * 0.15
        + f["interview_completion_rate"] * 0.15
        + f["offer_acceptance_rate"] * 0.10
        + f["github_activity"] * 0.20
        + f["avg_skill_assessment"] * 0.15
        + f["profile_views_norm"] * 0.05
        + f["saved_by_recruiters_norm"] * 0.05
        + f["verified"] * 0.025
        + f["linkedin_connected"] * 0.025
        - f["notice_penalty"] * 0.05
    )
    f["behavioral_score"] = round(max(0.0, min(b, 1.0)), 4)
    return f


def extract_all_features(candidate: dict) -> dict[str, Any]:
    """
    Master feature extractor. Returns a flat feature dict for one candidate.
    """
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    career_feats = extract_career_features(career, profile)
    edu_feats = extract_education_features(education)
    behavioral_feats = extract_behavioral_features(signals)
    skill_feats = compute_skill_score(skills)

    # YOE normalized
    yoe = float(profile.get("years_of_experience", 0) or 0)
    yoe_norm = min(yoe / 15.0, 1.0)

    # Location features
    location = normalize_text(profile.get("location", "") or "")
    country = normalize_text(profile.get("country", "") or "")
    is_india = "india" in country or "in" == country
    is_major_hub = any(
        city in location
        for city in ["bangalore", "bengaluru", "hyderabad", "mumbai",
                     "delhi", "pune", "chennai", "gurugram", "noida"]
    )

    honeypot_flags = candidate.get("honeypot_flags", {})
    honeypot_count = sum(honeypot_flags.values())

    return {
        "candidate_id": candidate.get("candidate_id"),
        # Raw scores
        "career_score": career_feats["career_score"],
        "skill_score": skill_feats["total_score"],
        "behavioral_score": behavioral_feats["behavioral_score"],
        "education_score": edu_feats["education_score"],
        # Sub-features (for explainability)
        "yoe_norm": yoe_norm,
        "years_of_experience": yoe,
        "production_signal_score": career_feats["production_signal_score"],
        "ml_signal_score": career_feats["ml_signal_score"],
        "product_company_score": career_feats["product_company_score"],
        "big_company_ratio": career_feats["big_company_ratio"],
        "total_career_months": career_feats["total_career_months"],
        "avg_tenure_months": career_feats["avg_tenure_months"],
        "job_hopping_penalty": career_feats["job_hopping_penalty"],
        "is_consulting_heavy": career_feats["is_consulting_heavy"],
        "skill_cluster_coverage": skill_feats["cluster_coverage"],
        "top_skills": skill_feats["top_skills"],
        "cluster_scores": skill_feats["cluster_scores"],
        "highest_degree_score": edu_feats["highest_degree_score"],
        "institution_tier_score": edu_feats["institution_tier_score"],
        "relevant_field": edu_feats["relevant_field"],
        "github_activity": behavioral_feats["github_activity"],
        "recruiter_responsiveness": behavioral_feats["recruiter_responsiveness"],
        "interview_completion_rate": behavioral_feats["interview_completion_rate"],
        "open_to_work": behavioral_feats["open_to_work"],
        "willing_to_relocate": behavioral_feats["willing_to_relocate"],
        "notice_period_days": behavioral_feats["notice_period_days"],
        "avg_skill_assessment": behavioral_feats["avg_skill_assessment"],
        "verified": behavioral_feats["verified"],
        # Honeypot
        "honeypot_flags": honeypot_flags,
        "honeypot_count": honeypot_count,
        "is_disqualified": candidate.get("is_disqualified", False),
        # Location
        "is_india": is_india,
        "is_major_hub": is_major_hub,
        "country": profile.get("country", ""),
        "location": profile.get("location", ""),
        "preferred_work_mode": (
            signals.get("preferred_work_mode", "flexible") or "flexible"
        ),
    }
