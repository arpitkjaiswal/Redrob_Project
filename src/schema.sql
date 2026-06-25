-- Task 3: Database Schema Design — SQLite Relational Schema
-- For teammate integration (backend APIs, database connectivity)
-- The ranking engine uses in-memory processing, but this schema
-- allows persistence and querying by the backend team.

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    anonymized_name TEXT NOT NULL,
    headline TEXT,
    summary TEXT,
    location TEXT,
    country TEXT,
    years_of_experience REAL DEFAULT 0,
    current_title TEXT,
    current_company TEXT,
    current_company_size TEXT,
    current_industry TEXT
);

CREATE TABLE IF NOT EXISTS career_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    company TEXT,
    title TEXT,
    start_date TEXT,
    end_date TEXT,
    duration_months INTEGER DEFAULT 0,
    is_current BOOLEAN DEFAULT 0,
    industry TEXT,
    company_size TEXT,
    description TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS education (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    institution TEXT,
    degree TEXT,
    field_of_study TEXT,
    start_year INTEGER,
    end_year INTEGER,
    grade TEXT,
    tier TEXT DEFAULT 'unknown',
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    name TEXT NOT NULL,
    proficiency TEXT DEFAULT 'beginner',
    endorsements INTEGER DEFAULT 0,
    duration_months INTEGER DEFAULT 0,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS certifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    name TEXT NOT NULL,
    issuer TEXT,
    year INTEGER,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS languages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    language TEXT NOT NULL,
    proficiency TEXT DEFAULT 'basic',
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS redrob_signals (
    candidate_id TEXT PRIMARY KEY,
    profile_completeness_score REAL DEFAULT 0,
    signup_date TEXT,
    last_active_date TEXT,
    open_to_work_flag BOOLEAN DEFAULT 0,
    profile_views_received_30d INTEGER DEFAULT 0,
    applications_submitted_30d INTEGER DEFAULT 0,
    recruiter_response_rate REAL DEFAULT 0,
    avg_response_time_hours REAL DEFAULT 0,
    skill_assessment_scores_json TEXT DEFAULT '{}',
    connection_count INTEGER DEFAULT 0,
    endorsements_received INTEGER DEFAULT 0,
    notice_period_days INTEGER DEFAULT 0,
    expected_salary_min_lpa REAL DEFAULT 0,
    expected_salary_max_lpa REAL DEFAULT 0,
    preferred_work_mode TEXT DEFAULT 'flexible',
    willing_to_relocate BOOLEAN DEFAULT 0,
    github_activity_score REAL DEFAULT -1,
    search_appearance_30d INTEGER DEFAULT 0,
    saved_by_recruiters_30d INTEGER DEFAULT 0,
    interview_completion_rate REAL DEFAULT 0,
    offer_acceptance_rate REAL DEFAULT -1,
    verified_email BOOLEAN DEFAULT 0,
    verified_phone BOOLEAN DEFAULT 0,
    linkedin_connected BOOLEAN DEFAULT 0,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

-- Pre-computed features for the AI pipeline
CREATE TABLE IF NOT EXISTS computed_features (
    candidate_id TEXT PRIMARY KEY,
    -- Career features
    relevant_ml_experience_months REAL DEFAULT 0,
    product_company_months REAL DEFAULT 0,
    consulting_only_flag BOOLEAN DEFAULT 0,
    title_relevance_score REAL DEFAULT 0,
    avg_tenure_months REAL DEFAULT 0,
    has_production_signals BOOLEAN DEFAULT 0,
    -- Skills features
    core_skill_match_count INTEGER DEFAULT 0,
    core_skill_weighted_score REAL DEFAULT 0,
    nice_to_have_skill_count INTEGER DEFAULT 0,
    skill_stuffer_score REAL DEFAULT 0,
    -- Education features
    education_tier_score REAL DEFAULT 0,
    relevant_degree_score REAL DEFAULT 0,
    -- Behavioral features
    availability_score REAL DEFAULT 0,
    engagement_score REAL DEFAULT 0,
    reliability_score REAL DEFAULT 0,
    -- Location features
    location_match_score REAL DEFAULT 0,
    -- Honeypot detection
    is_honeypot BOOLEAN DEFAULT 0,
    honeypot_confidence REAL DEFAULT 0,
    -- Final scores
    semantic_similarity_score REAL DEFAULT 0,
    final_composite_score REAL DEFAULT 0,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

-- Final ranking output
CREATE TABLE IF NOT EXISTS rankings (
    candidate_id TEXT PRIMARY KEY,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    reasoning TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_career_candidate ON career_history(candidate_id);
CREATE INDEX IF NOT EXISTS idx_skills_candidate ON skills(candidate_id);
CREATE INDEX IF NOT EXISTS idx_education_candidate ON education(candidate_id);
CREATE INDEX IF NOT EXISTS idx_computed_score ON computed_features(final_composite_score DESC);
