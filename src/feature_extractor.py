"""
Task 4: Feature Extraction

Extracts multi-dimensional features from candidate profiles:
- Career features (title relevance, experience, company type)
- Skills features (core match, trust scores)
- Education features (tier, degree relevance)
- Behavioral/signal features (availability, engagement, reliability)
- Location features
- Honeypot detection features
"""

import re
import math
import logging
from datetime import datetime, date
from typing import Optional

from .data_cleaner import parse_date_safe, detect_anomalies
from .skill_matcher import (
    match_skills_to_clusters,
    compute_skill_trust_score,
    detect_keyword_stuffer,
    PROFICIENCY_SCORES,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant", "cognizant technology solutions",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mindtree",  # now part of LTIMindtree
    "ltimindtree",
    "mphasis",
    "l&t infotech", "lti",
    "hexaware",
    "persistent systems",
    "cyient",
    "zensar",
}

RELEVANT_AI_TITLES = {
    "ai engineer": 1.0,
    "senior ai engineer": 1.0,
    "ml engineer": 1.0,
    "machine learning engineer": 1.0,
    "senior ml engineer": 1.0,
    "senior machine learning engineer": 1.0,
    "principal ml engineer": 0.95,
    "staff ml engineer": 0.95,
    "data scientist": 0.85,
    "senior data scientist": 0.85,
    "lead data scientist": 0.85,
    "applied scientist": 0.85,
    "nlp engineer": 0.9,
    "search engineer": 0.9,
    "ranking engineer": 0.95,
    "recommendation engineer": 0.9,
    "research engineer": 0.7,
    "ml researcher": 0.65,
    "research scientist": 0.6,
    "software engineer": 0.4,  # generic, needs context
    "senior software engineer": 0.45,
    "backend engineer": 0.35,
    "data engineer": 0.5,
    "senior data engineer": 0.5,
    "analytics engineer": 0.4,
    "junior ml engineer": 0.7,
    "junior data scientist": 0.65,
    "full stack developer": 0.2,
    "frontend engineer": 0.1,
    "devops engineer": 0.15,
}

# Non-relevant titles (strong negative signal)
NON_RELEVANT_TITLES = {
    "marketing manager", "hr manager", "accountant",
    "sales executive", "sales manager", "content writer",
    "graphic designer", "operations manager", "customer support",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "business analyst",  # somewhat negative
    "project manager",  # negative for this role
    "product manager",  # somewhat ok
    "teacher", "professor",
}

INDIA_TIER1_CITIES = {
    "mumbai", "delhi", "delhi ncr", "bangalore", "bengaluru",
    "hyderabad", "chennai", "kolkata", "pune", "noida",
    "gurgaon", "gurugram", "ghaziabad", "faridabad",
    "navi mumbai", "thane",
}

# JD preferred locations
JD_PREFERRED_LOCATIONS = {"pune", "noida"}
JD_ACCEPTABLE_LOCATIONS = {
    "hyderabad", "mumbai", "delhi", "delhi ncr",
    "bangalore", "bengaluru", "gurgaon", "gurugram",
}

RELEVANT_FIELDS_OF_STUDY = {
    "computer science": 1.0, "cs": 1.0,
    "machine learning": 1.0, "artificial intelligence": 1.0,
    "data science": 0.95,
    "information technology": 0.8, "it": 0.8,
    "software engineering": 0.85,
    "statistics": 0.75, "applied statistics": 0.75,
    "mathematics": 0.7, "applied mathematics": 0.7,
    "electronics": 0.5, "electrical engineering": 0.5,
    "electronics and communication": 0.5,
    "information systems": 0.6,
    "computational linguistics": 0.8,
}


def extract_all_features(data: dict) -> dict:
    """
    Extract ALL features from a candidate dict.
    Returns a flat dict of feature_name -> value.
    """
    features = {}

    profile = data.get("profile", {})
    career = data.get("career_history", [])
    education = data.get("education", [])
    skills = data.get("skills", [])
    certs = data.get("certifications", [])
    signals = data.get("redrob_signals", {})

    # ─── Career Features ────────────────────────────────────────────────
    features.update(_extract_career_features(profile, career))

    # ─── Skills Features ────────────────────────────────────────────────
    features.update(_extract_skills_features(skills, career, profile, signals))

    # ─── Education Features ─────────────────────────────────────────────
    features.update(_extract_education_features(education))

    # ─── Behavioral/Signal Features ─────────────────────────────────────
    features.update(_extract_behavioral_features(signals))

    # ─── Location Features ──────────────────────────────────────────────
    features.update(_extract_location_features(profile, signals))

    # ─── Honeypot Detection ─────────────────────────────────────────────
    features.update(_detect_honeypot_features(data))

    return features


def _extract_career_features(profile: dict, career: list) -> dict:
    """Extract career-related features."""
    features = {}

    # Total experience
    yoe = profile.get("years_of_experience", 0)
    features["total_experience_years"] = float(yoe)

    # Experience fit (5-9 years ideal, 6-8 ideal sweet spot)
    if 6 <= yoe <= 8:
        features["experience_fit_score"] = 1.0
    elif 5 <= yoe <= 9:
        features["experience_fit_score"] = 0.85
    elif 4 <= yoe <= 12:
        features["experience_fit_score"] = 0.6
    elif 3 <= yoe <= 15:
        features["experience_fit_score"] = 0.3
    else:
        features["experience_fit_score"] = 0.1

    # Current title relevance
    current_title = profile.get("current_title", "").lower().strip()
    features["current_title_relevance"] = RELEVANT_AI_TITLES.get(current_title, 0.0)

    # Check if current title is explicitly non-relevant
    features["is_non_relevant_title"] = 1.0 if current_title in NON_RELEVANT_TITLES else 0.0

    # Best title relevance across career
    best_title_score = features["current_title_relevance"]
    for entry in career:
        title_lower = entry.get("title", "").lower().strip()
        title_score = RELEVANT_AI_TITLES.get(title_lower, 0.0)
        best_title_score = max(best_title_score, title_score)
    features["best_title_relevance"] = best_title_score

    # ML/AI experience estimation from career descriptions
    ml_months = 0
    for entry in career:
        desc = entry.get("description", "").lower()
        title = entry.get("title", "").lower()
        duration = entry.get("duration_months", 0)

        # Strong AI/ML indicators in description
        strong_ai_keywords = [
            "machine learning", "deep learning", "neural network",
            "model training", "model deployment", "embeddings",
            "nlp", "natural language processing", "recommendation system",
            "ranking system", "search system", "data science",
            "ml pipeline", "ml model", "ai system", "vector",
            "transformer", "bert", "gpt", "fine-tun",
            "retrieval", "classification model", "prediction model",
        ]
        ai_mention_count = sum(1 for kw in strong_ai_keywords if kw in desc)

        # Title-based detection
        title_is_ai = RELEVANT_AI_TITLES.get(title, 0.0) >= 0.6

        if title_is_ai or ai_mention_count >= 3:
            ml_months += duration
        elif ai_mention_count >= 1:
            ml_months += duration * 0.5

    features["estimated_ml_experience_months"] = ml_months
    features["estimated_ml_experience_years"] = ml_months / 12.0

    # Product company vs consulting detection
    consulting_months = 0
    product_months = 0
    total_career_months = 0
    for entry in career:
        company = entry.get("company", "").lower().strip()
        duration = entry.get("duration_months", 0)
        total_career_months += duration

        if company in CONSULTING_FIRMS:
            consulting_months += duration
        else:
            product_months += duration

    features["consulting_months"] = consulting_months
    features["product_company_months"] = product_months
    features["total_career_months"] = total_career_months

    # Consulting-only flag (JD explicit disqualifier)
    features["consulting_only_flag"] = (
        1.0 if total_career_months > 0 and consulting_months == total_career_months
        else 0.0
    )

    # Product company ratio
    features["product_company_ratio"] = (
        product_months / total_career_months if total_career_months > 0 else 0.0
    )

    # Average tenure (detect title-chasers with <18 months avg)
    num_roles = len(career)
    if num_roles > 0:
        avg_tenure = total_career_months / num_roles
        features["avg_tenure_months"] = avg_tenure
        features["is_title_chaser"] = 1.0 if avg_tenure < 18 and num_roles >= 3 else 0.0
    else:
        features["avg_tenure_months"] = 0
        features["is_title_chaser"] = 0.0

    # Number of companies
    companies = set(e.get("company", "").lower() for e in career if e.get("company"))
    features["num_companies"] = len(companies)

    # Has current role
    features["has_current_role"] = 1.0 if any(e.get("is_current") for e in career) else 0.0

    # Production deployment signals from career descriptions
    all_descs = " ".join(e.get("description", "") for e in career).lower()
    production_keywords = [
        "production", "deployed", "shipped", "launched", "live",
        "real users", "scale", "serving", "api", "endpoint",
        "million users", "real-time", "latency", "throughput",
        "monitoring", "alerting", "on-call",
    ]
    features["production_signal_count"] = sum(
        1 for kw in production_keywords if kw in all_descs
    )
    features["has_production_signals"] = 1.0 if features["production_signal_count"] >= 2 else 0.0

    return features


def _extract_skills_features(
    skills: list, career: list, profile: dict, signals: dict
) -> dict:
    """Extract skills-related features using cluster matching."""
    features = {}

    # Get career descriptions
    career_descriptions = [e.get("description", "") for e in career]

    # Skills as dicts
    skill_dicts = [
        {
            "name": s.get("name", ""),
            "proficiency": s.get("proficiency", "beginner"),
            "endorsements": s.get("endorsements", 0),
            "duration_months": s.get("duration_months", 0),
        }
        for s in skills
    ]

    # Cluster matching
    cluster_results = match_skills_to_clusters(skill_dicts, career_descriptions)

    # Must-have clusters score
    must_have_clusters = [
        "embeddings_retrieval", "vector_databases", "python_strong",
        "nlp_ir", "ranking_evaluation", "ranking_recommendation_systems",
    ]
    must_have_matched = 0
    must_have_total_score = 0.0
    for cluster_name in must_have_clusters:
        result = cluster_results.get(cluster_name, {})
        if result.get("matched"):
            must_have_matched += 1
            must_have_total_score += result["score"] * result["weight"]

    features["must_have_clusters_matched"] = must_have_matched
    features["must_have_clusters_score"] = must_have_total_score / len(must_have_clusters)

    # Nice-to-have clusters
    nice_to_have_clusters = [
        "llm_finetuning", "ml_production", "data_engineering", "deep_learning_core",
    ]
    nice_matched = 0
    nice_total_score = 0.0
    for cluster_name in nice_to_have_clusters:
        result = cluster_results.get(cluster_name, {})
        if result.get("matched"):
            nice_matched += 1
            nice_total_score += result["score"] * result["weight"]

    features["nice_to_have_matched"] = nice_matched
    features["nice_to_have_score"] = nice_total_score / max(1, len(nice_to_have_clusters))

    # Individual cluster scores (for reasoning)
    for cluster_name, result in cluster_results.items():
        features[f"cluster_{cluster_name}_matched"] = 1.0 if result["matched"] else 0.0
        features[f"cluster_{cluster_name}_score"] = result["score"]

    # Skill trust scores
    assessment_scores = signals.get("skill_assessment_scores", {})
    trust_scores = []
    for s in skill_dicts:
        trust = compute_skill_trust_score(s, assessment_scores)
        trust_scores.append(trust)

    features["avg_skill_trust_score"] = (
        sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
    )

    # Keyword stuffer detection
    current_title = profile.get("current_title", "")
    features["keyword_stuffer_score"] = detect_keyword_stuffer(
        skill_dicts, career, current_title
    )

    # Total skills count
    features["total_skills_count"] = len(skills)

    # Expert skills count
    features["expert_skills_count"] = sum(
        1 for s in skills if s.get("proficiency") == "expert"
    )
    features["advanced_skills_count"] = sum(
        1 for s in skills if s.get("proficiency") == "advanced"
    )

    # Relevant certifications
    relevant_cert_issuers = {
        "aws", "google", "microsoft", "nvidia", "deeplearning.ai",
        "coursera", "stanford", "mit",
    }
    # Certs not passed to this function - handle via data dict
    features["has_relevant_certs"] = 0.0  # Will be updated if we add certs

    return features


def _extract_education_features(education: list) -> dict:
    """Extract education-related features."""
    features = {}

    if not education:
        features["education_tier_score"] = 0.0
        features["relevant_degree_score"] = 0.0
        features["has_advanced_degree"] = 0.0
        features["best_tier"] = "unknown"
        return features

    # Best institution tier
    tier_scores = {
        "tier_1": 1.0,
        "tier_2": 0.7,
        "tier_3": 0.4,
        "tier_4": 0.2,
        "unknown": 0.1,
    }
    best_tier = "unknown"
    best_tier_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        score = tier_scores.get(tier, 0.1)
        if score > best_tier_score:
            best_tier_score = score
            best_tier = tier

    features["education_tier_score"] = best_tier_score
    features["best_tier"] = best_tier

    # Relevant field of study
    best_field_score = 0.0
    for edu in education:
        field = edu.get("field_of_study", "").lower().strip()
        field_score = RELEVANT_FIELDS_OF_STUDY.get(field, 0.0)
        # Also check partial matches
        if field_score == 0.0:
            for key, val in RELEVANT_FIELDS_OF_STUDY.items():
                if key in field or field in key:
                    field_score = max(field_score, val * 0.8)
        best_field_score = max(best_field_score, field_score)

    features["relevant_degree_score"] = best_field_score

    # Advanced degree
    advanced_degrees = {"m.tech", "m.e.", "m.sc", "m.s.", "ph.d", "phd", "mba"}
    has_advanced = any(
        edu.get("degree", "").lower().strip() in advanced_degrees
        for edu in education
    )
    features["has_advanced_degree"] = 1.0 if has_advanced else 0.0

    return features


def _extract_behavioral_features(signals: dict) -> dict:
    """Extract behavioral features from Redrob signals."""
    features = {}

    # ─── Availability Score ──────────────────────────────────────────────
    # How available/reachable is this candidate?

    # Recency of activity
    last_active = parse_date_safe(signals.get("last_active_date", ""))
    if last_active:
        days_since_active = (date.today() - last_active).days
        if days_since_active <= 7:
            recency_score = 1.0
        elif days_since_active <= 30:
            recency_score = 0.9
        elif days_since_active <= 90:
            recency_score = 0.7
        elif days_since_active <= 180:
            recency_score = 0.4
        else:
            recency_score = 0.1
    else:
        recency_score = 0.0

    features["recency_score"] = recency_score
    features["open_to_work"] = 1.0 if signals.get("open_to_work_flag") else 0.0

    # Recruiter response rate (very important per JD)
    response_rate = signals.get("recruiter_response_rate", 0)
    features["recruiter_response_rate"] = response_rate

    # Response time
    avg_response_time = signals.get("avg_response_time_hours", 999)
    if avg_response_time <= 12:
        response_time_score = 1.0
    elif avg_response_time <= 24:
        response_time_score = 0.9
    elif avg_response_time <= 48:
        response_time_score = 0.7
    elif avg_response_time <= 96:
        response_time_score = 0.5
    else:
        response_time_score = 0.2
    features["response_time_score"] = response_time_score

    # Composite availability
    features["availability_score"] = (
        0.30 * recency_score
        + 0.25 * features["open_to_work"]
        + 0.30 * response_rate
        + 0.15 * response_time_score
    )

    # ─── Engagement Score ────────────────────────────────────────────────
    profile_completeness = signals.get("profile_completeness_score", 0) / 100.0
    features["profile_completeness"] = profile_completeness

    connection_count = min(signals.get("connection_count", 0), 500)
    connection_score = connection_count / 500.0
    features["connection_score"] = connection_score

    endorsements = min(signals.get("endorsements_received", 0), 100)
    endorsement_score = endorsements / 100.0
    features["endorsement_score"] = endorsement_score

    features["engagement_score"] = (
        0.40 * profile_completeness
        + 0.30 * connection_score
        + 0.30 * endorsement_score
    )

    # ─── Reliability Score ───────────────────────────────────────────────
    interview_rate = signals.get("interview_completion_rate", 0)
    features["interview_completion_rate"] = interview_rate

    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate < 0:
        offer_score = 0.5  # Unknown, neutral
    else:
        offer_score = offer_rate
    features["offer_acceptance_score"] = offer_score

    features["reliability_score"] = (
        0.60 * interview_rate
        + 0.40 * offer_score
    )

    # ─── Technical Activity ──────────────────────────────────────────────
    github = signals.get("github_activity_score", -1)
    if github < 0:
        github_score = 0.0  # No GitHub linked
    else:
        github_score = github / 100.0
    features["github_activity_score"] = github_score

    # ─── Market Signal ───────────────────────────────────────────────────
    # Are other recruiters interested in this person?
    saved = min(signals.get("saved_by_recruiters_30d", 0), 30)
    features["saved_by_recruiters_score"] = saved / 30.0

    search_appearances = min(signals.get("search_appearance_30d", 0), 300)
    features["search_appearance_score"] = search_appearances / 300.0

    # ─── Verification ────────────────────────────────────────────────────
    verified_count = (
        (1 if signals.get("verified_email") else 0)
        + (1 if signals.get("verified_phone") else 0)
        + (1 if signals.get("linkedin_connected") else 0)
    )
    features["verification_score"] = verified_count / 3.0

    # ─── Notice Period ───────────────────────────────────────────────────
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 45:
        notice_score = 0.85
    elif notice_days <= 60:
        notice_score = 0.65
    elif notice_days <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.15
    features["notice_period_score"] = notice_score
    features["notice_period_days"] = notice_days

    # ─── Work Mode Match ────────────────────────────────────────────────
    work_mode = signals.get("preferred_work_mode", "flexible")
    # JD says hybrid-flexible
    work_mode_scores = {
        "hybrid": 1.0,
        "flexible": 0.95,
        "onsite": 0.7,
        "remote": 0.5,
    }
    features["work_mode_score"] = work_mode_scores.get(work_mode, 0.5)

    return features


def _extract_location_features(profile: dict, signals: dict) -> dict:
    """Extract location-related features."""
    features = {}

    location = profile.get("location", "").lower().strip()
    country = profile.get("country", "").lower().strip()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # Country match
    is_india = country == "india"
    features["is_india"] = 1.0 if is_india else 0.0

    # City match
    location_parts = [p.strip() for p in re.split(r'[,/]', location)]
    location_lower_parts = [p.lower() for p in location_parts]

    is_preferred = any(
        city in loc for loc in location_lower_parts
        for city in JD_PREFERRED_LOCATIONS
    )
    is_acceptable = any(
        city in loc for loc in location_lower_parts
        for city in JD_ACCEPTABLE_LOCATIONS
    )
    is_tier1_india = any(
        city in loc for loc in location_lower_parts
        for city in INDIA_TIER1_CITIES
    )

    if is_preferred:
        location_score = 1.0
    elif is_acceptable:
        location_score = 0.85
    elif is_tier1_india:
        location_score = 0.7
    elif is_india:
        location_score = 0.5
    elif willing_to_relocate:
        location_score = 0.3
    else:
        location_score = 0.15

    features["location_match_score"] = location_score
    features["willing_to_relocate"] = 1.0 if willing_to_relocate else 0.0

    return features


def _detect_honeypot_features(data: dict) -> dict:
    """
    Detect honeypot candidates with impossible profiles.
    Returns honeypot probability features.
    """
    features = {}
    anomalies = detect_anomalies(data)

    profile = data.get("profile", {})
    career = data.get("career_history", [])
    skills = data.get("skills", [])
    signals = data.get("redrob_signals", {})

    honeypot_score = 0.0

    # 1. Impossible experience duration
    stated_yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(e.get("duration_months", 0) for e in career)
    career_yoe = total_career_months / 12.0

    if stated_yoe > 0 and career_yoe > 0:
        yoe_ratio = abs(stated_yoe - career_yoe) / max(stated_yoe, 1)
        if yoe_ratio > 0.5:
            honeypot_score += 0.25

    # 2. Expert proficiency with 0 duration
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 5:
        honeypot_score += 0.35
    elif expert_zero >= 3:
        honeypot_score += 0.2

    # 3. Too many expert skills
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count > 10:
        honeypot_score += 0.3
    elif expert_count > 8:
        honeypot_score += 0.15

    # 4. Career date impossibilities (duration_months doesn't match dates)
    for entry in career:
        start = parse_date_safe(entry.get("start_date"))
        end = parse_date_safe(entry.get("end_date"))
        stated_duration = entry.get("duration_months", 0)

        if start and (end or entry.get("is_current")):
            if end is None:
                end = date.today()
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            if actual_months >= 0 and stated_duration > 0:
                if stated_duration > actual_months * 2 + 6:
                    honeypot_score += 0.2
                    break

    # 5. Title-description severe mismatch across ALL career entries
    mismatch_count = 0
    for entry in career:
        title = entry.get("title", "").lower()
        desc = entry.get("description", "").lower()

        # Completely unrelated title-description pairs
        title_is_marketing = "marketing" in title
        title_is_hr = "hr" in title or "human resource" in title
        title_is_accounting = "account" in title
        title_is_mechanical = "mechanical" in title
        title_is_civil = "civil" in title

        desc_is_ml = any(kw in desc for kw in [
            "machine learning", "deep learning", "model training",
            "neural network", "embeddings", "nlp"
        ])
        desc_is_software = any(kw in desc for kw in [
            "code", "software", "api", "database", "pipeline",
            "deploy", "infrastructure"
        ])

        if (title_is_marketing or title_is_hr or title_is_accounting) and desc_is_ml:
            mismatch_count += 1
        if (title_is_mechanical or title_is_civil) and desc_is_ml:
            mismatch_count += 1

    if mismatch_count >= 2:
        honeypot_score += 0.2

    # 6. Anomaly count from cleaner
    if len(anomalies) >= 4:
        honeypot_score += 0.15

    features["honeypot_score"] = min(1.0, honeypot_score)
    features["is_likely_honeypot"] = 1.0 if honeypot_score >= 0.5 else 0.0
    features["anomaly_count"] = len(anomalies)

    return features
