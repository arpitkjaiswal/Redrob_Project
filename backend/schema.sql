-- Redrob Candidate Ranking Engine — SQLite Schema

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id        TEXT PRIMARY KEY,
    anonymized_name     TEXT,
    headline            TEXT,
    summary             TEXT,
    location            TEXT,
    country             TEXT,
    years_of_experience REAL,
    current_title       TEXT,
    current_company     TEXT,
    current_company_size TEXT,
    current_industry    TEXT,
    open_to_work        INTEGER DEFAULT 0,
    willing_to_relocate INTEGER DEFAULT 0,
    preferred_work_mode TEXT,
    notice_period_days  INTEGER,
    expected_salary_min REAL,
    expected_salary_max REAL,
    github_activity_score REAL DEFAULT -1,
    profile_completeness_score REAL DEFAULT 0,
    linkedin_connected   INTEGER DEFAULT 0,
    verified_email       INTEGER DEFAULT 0,
    verified_phone       INTEGER DEFAULT 0,
    last_active_date     TEXT,
    signup_date          TEXT,
    raw_json             TEXT   -- full candidate JSON blob
);

CREATE TABLE IF NOT EXISTS candidate_skills (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id   TEXT NOT NULL REFERENCES candidates(candidate_id),
    skill_name     TEXT NOT NULL,
    proficiency    TEXT,
    endorsements   INTEGER DEFAULT 0,
    duration_months INTEGER DEFAULT 0,
    cluster        TEXT
);

CREATE TABLE IF NOT EXISTS candidate_education (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id   TEXT NOT NULL REFERENCES candidates(candidate_id),
    institution    TEXT,
    degree         TEXT,
    field_of_study TEXT,
    start_year     INTEGER,
    end_year       INTEGER,
    grade          TEXT,
    tier           TEXT
);

CREATE TABLE IF NOT EXISTS candidate_career (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id    TEXT NOT NULL REFERENCES candidates(candidate_id),
    company         TEXT,
    title           TEXT,
    start_date      TEXT,
    end_date        TEXT,
    duration_months INTEGER,
    is_current      INTEGER DEFAULT 0,
    industry        TEXT,
    company_size    TEXT,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS rankings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL,
    candidate_id     TEXT NOT NULL REFERENCES candidates(candidate_id),
    rank             INTEGER,
    score            REAL,
    career_score     REAL,
    skill_score      REAL,
    behavioral_score REAL,
    education_score  REAL,
    embedding_sim    REAL,
    is_honeypot      INTEGER DEFAULT 0,
    reasoning        TEXT,
    created_at       TEXT DEFAULT (datetime('now')),
    UNIQUE(job_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    title        TEXT,
    description  TEXT,
    created_at   TEXT DEFAULT (datetime('now')),
    status       TEXT DEFAULT 'pending'  -- pending | running | done | failed
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id       TEXT PRIMARY KEY,
    job_id       TEXT REFERENCES jobs(job_id),
    status       TEXT DEFAULT 'running',
    started_at   TEXT DEFAULT (datetime('now')),
    finished_at  TEXT,
    total_candidates INTEGER,
    error_message TEXT
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_skills_candidate   ON candidate_skills(candidate_id);
CREATE INDEX IF NOT EXISTS idx_career_candidate   ON candidate_career(candidate_id);
CREATE INDEX IF NOT EXISTS idx_education_candidate ON candidate_education(candidate_id);
CREATE INDEX IF NOT EXISTS idx_rankings_job       ON rankings(job_id, rank);
CREATE INDEX IF NOT EXISTS idx_rankings_score     ON rankings(job_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_country ON candidates(country);
CREATE INDEX IF NOT EXISTS idx_candidates_yoe     ON candidates(years_of_experience);
