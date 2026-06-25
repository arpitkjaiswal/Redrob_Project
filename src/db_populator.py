import sqlite3
import json
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

def setup_database(db_path: str, schema_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    with open(schema_path, "r") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    return conn

def populate_candidates(conn: sqlite3.Connection, candidates: List[Dict]):
    cursor = conn.cursor()
    cands_data, career_data, edu_data, skills_data, certs_data, lang_data, signals_data = [], [], [], [], [], [], []

    for c in candidates:
        cid = c["candidate_id"]
        profile = c.get("profile", {})
        signals = c.get("redrob_signals", {})

        cands_data.append((
            cid, c.get("anonymized_name", ""), profile.get("headline"), profile.get("summary"),
            profile.get("location"), profile.get("country"), profile.get("years_of_experience", 0),
            profile.get("current_title"), profile.get("current_company"), profile.get("current_company_size"),
            profile.get("current_industry")
        ))

        for entry in c.get("career_history", []):
            career_data.append((
                cid, entry.get("company"), entry.get("title"), entry.get("start_date"),
                entry.get("end_date"), entry.get("duration_months", 0), entry.get("is_current", False),
                entry.get("industry"), entry.get("company_size"), entry.get("description")
            ))

        for edu in c.get("education", []):
            edu_data.append((
                cid, edu.get("institution"), edu.get("degree"), edu.get("field_of_study"),
                edu.get("start_year"), edu.get("end_year"), edu.get("grade"), edu.get("tier", "unknown")
            ))

        for skill in c.get("skills", []):
            skills_data.append((
                cid, skill.get("name"), skill.get("proficiency", "beginner"),
                skill.get("endorsements", 0), skill.get("duration_months", 0)
            ))

        for cert in c.get("certifications", []):
            certs_data.append((cid, cert.get("name"), cert.get("issuer"), cert.get("year")))

        for lang in c.get("languages", []):
            lang_data.append((cid, lang.get("language"), lang.get("proficiency", "basic")))

        signals_data.append((
            cid, signals.get("profile_completeness_score", 0), signals.get("signup_date"),
            signals.get("last_active_date"), signals.get("open_to_work_flag", False),
            signals.get("profile_views_received_30d", 0), signals.get("applications_submitted_30d", 0),
            signals.get("recruiter_response_rate", 0), signals.get("avg_response_time_hours", 0),
            json.dumps(signals.get("skill_assessment_scores", {})), signals.get("connection_count", 0),
            signals.get("endorsements_received", 0), signals.get("notice_period_days", 0),
            signals.get("expected_salary_min_lpa", 0), signals.get("expected_salary_max_lpa", 0),
            signals.get("preferred_work_mode", "flexible"), signals.get("willing_to_relocate", False),
            signals.get("github_activity_score", -1), signals.get("search_appearance_30d", 0),
            signals.get("saved_by_recruiters_30d", 0), signals.get("interview_completion_rate", 0),
            signals.get("offer_acceptance_rate", -1), signals.get("verified_email", False),
            signals.get("verified_phone", False), signals.get("linkedin_connected", False)
        ))

    # Disable synchronous for faster bulk insert
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")

    cursor.executemany("INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", cands_data)
    cursor.executemany("INSERT INTO career_history (candidate_id, company, title, start_date, end_date, duration_months, is_current, industry, company_size, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", career_data)
    cursor.executemany("INSERT INTO education (candidate_id, institution, degree, field_of_study, start_year, end_year, grade, tier) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", edu_data)
    cursor.executemany("INSERT INTO skills (candidate_id, name, proficiency, endorsements, duration_months) VALUES (?, ?, ?, ?, ?)", skills_data)
    cursor.executemany("INSERT INTO certifications (candidate_id, name, issuer, year) VALUES (?, ?, ?, ?)", certs_data)
    cursor.executemany("INSERT INTO languages (candidate_id, language, proficiency) VALUES (?, ?, ?)", lang_data)
    cursor.executemany("INSERT OR REPLACE INTO redrob_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", signals_data)

    conn.commit()
    logger.info(f"Populated SQLite database with {len(cands_data)} candidates.")

def populate_computed_features(conn: sqlite3.Connection, features_dict: Dict[str, Dict]):
    cursor = conn.cursor()
    data = []
    for cid, f in features_dict.items():
        data.append((
            cid, f.get("estimated_ml_experience_months", 0), f.get("product_company_months", 0),
            f.get("consulting_only_flag", 0), f.get("best_title_relevance", 0), f.get("avg_tenure_months", 0),
            f.get("has_production_signals", 0), f.get("must_have_clusters_matched", 0),
            f.get("must_have_clusters_score", 0), f.get("nice_to_have_matched", 0),
            f.get("keyword_stuffer_score", 0), f.get("education_tier_score", 0),
            f.get("relevant_degree_score", 0), f.get("availability_score", 0),
            f.get("engagement_score", 0), f.get("reliability_score", 0),
            f.get("location_match_score", 0), f.get("is_likely_honeypot", 0),
            f.get("honeypot_score", 0), 0, 0
        ))

    cursor.executemany("""
    INSERT OR REPLACE INTO computed_features (
        candidate_id, relevant_ml_experience_months, product_company_months, consulting_only_flag,
        title_relevance_score, avg_tenure_months, has_production_signals, core_skill_match_count,
        core_skill_weighted_score, nice_to_have_skill_count, skill_stuffer_score, education_tier_score,
        relevant_degree_score, availability_score, engagement_score, reliability_score, location_match_score,
        is_honeypot, honeypot_confidence, semantic_similarity_score, final_composite_score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    logger.info(f"Populated computed features for {len(data)} candidates.")

def update_rankings(db_path: str, rankings: List[Dict]):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    data = []
    for c in rankings:
        data.append((c["candidate_id"], c["rank"], c["score"], c.get("reasoning", "")))
        
        # Also update final_composite_score and semantic_similarity_score in computed_features
        component_scores = c.get("component_scores", {})
        cursor.execute("UPDATE computed_features SET final_composite_score = ?, semantic_similarity_score = ? WHERE candidate_id = ?",
                       (c["score"], component_scores.get("semantic_similarity", 0), c["candidate_id"]))

    cursor.executemany("INSERT OR REPLACE INTO rankings (candidate_id, rank, score, reasoning) VALUES (?, ?, ?, ?)", data)
    conn.commit()
    conn.close()
    logger.info(f"Updated SQLite database with {len(rankings)} rankings.")
