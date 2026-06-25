# Redrob AI Candidate Ranking Engine

An intelligent, hybrid AI ranking engine designed to evaluate and rank 100K candidates for a Senior AI Engineer role. Built for the "Data & AI Challenge", this engine goes beyond simple keyword matching by semantically understanding career trajectories, production experience signals, and behavioral patterns to surface the genuinely best-fit candidates.

## Features

*   **Semantic Skill Clusters:** Matches skills conceptually (e.g., matching "FAISS" to the `vector_databases` cluster) rather than requiring exact keyword hits.
*   **Trust-Weighted Skill Scoring:** Validates skills using endorsements, stated proficiency, and duration. Penalizes keyword stuffers who list "expert" skills without any actual experience.
*   **Career Context Mining:** Analyzes unstructured career descriptions to find evidence of relevant ML work and production deployments, even if they aren't explicitly listed in the skills array.
*   **Honeypot & Anomaly Detection:** Identifies misleading profiles by catching:
    *   Expert proficiency with 0 duration months.
    *   Time-traveling career dates (stated duration > possible time).
    *   Title-description mismatches.
*   **Hybrid Scoring Model:** Combines TF-IDF/Sentence-Transformers text embeddings with weighted feature extraction (Career, Skills, Behavioral, Education, Location) for robust ranking.
*   **Personalized Reasoning:** Automatically generates a human-readable justification for why a candidate received their specific score.
*   **SQLite Integration:** Pre-computes all features and rankings into a `redrob_candidates.db` SQLite database for seamless integration with backend APIs.
*   **REST API:** A full FastAPI backend exposing candidates, rankings, and pipeline management over HTTP.

---

## Architecture

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────────────────┐
│                  src/ ML Pipeline                │
│                                                 │
│  data_loader → data_cleaner → feature_extractor │
│       ↓                                         │
│  embedding_engine  (TF-IDF / sentence-transformers)
│       ↓                                         │
│  ranking_engine  (hybrid composite score)       │
│       ↓                                         │
│  reasoning_generator  (human-readable text)     │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   submission.csv           SQLite DB
   (challenge output)  (redrob_candidates.db)
                              │
                              ▼
                  ┌───────────────────────┐
                  │   backend/ FastAPI     │
                  │   REST API on :8000   │
                  └───────────────────────┘
```

**Pipeline stages:**
1. **Data Loading & Cleaning:** Stream-loads the 100K JSONL file, normalizing text and filtering anomalies.
2. **Feature Extraction:** Extracts over 50 numerical and boolean features across career history, education, and behavioral signals.
3. **Embedding Generation:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to encode the job description and candidate profiles, computing cosine similarities.
4. **Hybrid Engine:** Merges embedding similarity with feature scores, applying hard penalties for disqualifiers and honeypots.
5. **Persistence:** Saves pre-computed artifacts and an SQLite database.
6. **Inference:** Rapidly ranks the top candidates within seconds.

---

## Installation & Usage

### Prerequisites
*   Python 3.11+
*   (Optional but recommended) NVIDIA GPU for faster `sentence-transformers` embedding generation.

### 1. Install Dependencies

```bash
# ML pipeline dependencies
pip install -r requirements.txt

# Backend API dependencies
pip install fastapi "uvicorn[standard]" pydantic
```

### 2. Pre-computation (One-time Setup)

Run the pre-computation pipeline. This will process the dataset, generate embeddings, and populate the SQLite database (`./precomputed/redrob_candidates.db`). *This step runs outside the 5-minute ranking window.*

```bash
python -m src.precompute --candidates ./candidates.jsonl --output ./precomputed --sqlite
```

### 3. Generate Rankings (CLI)

Run the incredibly fast ranking step. This uses the pre-computed artifacts to rank the candidates and output the final `submission.csv` with justifications.

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --precomputed-dir ./precomputed --analyze
```

### 4. Validate Submission

Verify that the output format strictly adheres to the challenge specifications:

```bash
python validate_submission.py submission.csv
```

---

## REST API Backend

The `backend/` package provides a full FastAPI server that exposes the ranking engine over HTTP.

### Start the API Server

```bash
# Run from the project root directory
python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI where you can test all endpoints directly in the browser.

### API Endpoints

#### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |

#### Candidates
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/candidates` | List candidates — paginated, filterable by `country`, `open_to_work`, `min_yoe`, `max_yoe` |
| `GET` | `/api/candidates/{candidate_id}` | Full profile with career history, education, and skills |
| `GET` | `/api/candidates/{candidate_id}/features` | Run live ML feature extraction (50+ features) for any candidate |

#### Jobs
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/jobs` | List all job postings |
| `POST` | `/api/jobs` | Create a new job posting |
| `GET` | `/api/jobs/{job_id}` | Get a single job |

#### Rankings
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/jobs/{job_id}/rankings` | Paginated rankings — filterable by `min_score`, `honeypots_only` |
| `GET` | `/api/jobs/{job_id}/rankings/{candidate_id}` | Score breakdown for one candidate in a job |

#### Pipeline (Trigger ML Ranking)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/pipeline/run` | Kick off a full ML ranking pipeline (runs async, returns `run_id`) |
| `GET` | `/api/pipeline/runs` | List all pipeline runs |
| `GET` | `/api/pipeline/runs/{run_id}` | Poll run status: `running` → `done` / `failed` |

### Typical API Workflow

```bash
# 1. Create a job posting
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_id": "senior-ai-eng-2024", "title": "Senior AI Engineer", "description": "..."}'

# 2. Trigger the ML pipeline (runs in background)
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"job_id": "senior-ai-eng-2024", "candidates_path": "./candidates.jsonl", "top_k": 100}'

# 3. Poll until done
curl http://localhost:8000/api/pipeline/runs/{run_id}

# 4. Fetch the ranked results
curl http://localhost:8000/api/jobs/senior-ai-eng-2024/rankings
```

---

## Evaluation Results

The ranking algorithm successfully surfaces the exact profile described in the Job Description. The Top 10 results yield:
*   **100% Product-Company background** (Engineers from Netflix, Flipkart, Microsoft, Amazon, etc.)
*   **Average ML/AI Experience:** 5.85 years
*   **0 Honeypots** or consulting-only mismatches in the Top 100.
*   **Ranking Inference Time:** ~2.7 seconds on CPU (well under the 5-minute limit).

---

##DEMO LINK

* https://redrob-project-ngxl.vercel.app/

---

## Project Structure

```
ai-candidate-ranking-engine/
│
├── rank.py                        # Main CLI ranking entry point
├── validate_submission.py         # Submission format validator
├── requirements.txt               # ML pipeline dependencies
├── candidate_schema.json          # Candidate data schema definition
├── sample_submission.csv          # Example output format
│
├── src/                           # ML pipeline
│   ├── __init__.py
│   ├── precompute.py              # One-time pre-computation entry point
│   ├── data_loader.py             # JSONL streaming and parsing
│   ├── data_cleaner.py            # Text normalization and honeypot detection
│   ├── data_preparation.py        # End-to-end data prep orchestrator
│   ├── feature_extractor.py       # 50+ feature derivations
│   ├── embedding_engine.py        # Cosine similarity and sentence-transformers
│   ├── skill_matcher.py           # Semantic clusters and trust-weighting
│   ├── ranking_engine.py          # Hybrid scoring logic
│   ├── reasoning_generator.py     # Natural language justifications
│   ├── db_populator.py            # SQLite persistence (src schema)
│   ├── evaluator.py               # Ranking quality analysis
│   ├── tune_weights.py            # Weight tuning utilities
│   ├── schema.py                  # Python schema definitions
│   └── schema.sql                 # src-side SQLite table definitions
│
├── backend/                       # FastAPI REST API
│   ├── __init__.py
│   ├── main.py                    # FastAPI app + all route definitions
│   ├── database.py                # SQLite connection manager
│   ├── schemas.py                 # Pydantic request/response models
│   ├── crud.py                    # All DB read/write operations
│   ├── pipeline.py                # Background ML pipeline runner
│   ├── schema.sql                 # Backend DB schema (jobs, pipeline_runs, etc.)
│   └── requirements.txt           # fastapi, uvicorn, pydantic
│
└── precomputed/                   # Generated at runtime (git-ignored)
    ├── redrob_candidates.db       # SQLite database
    ├── features.pkl               # Pre-computed feature vectors
    ├── similarities.npy           # Pre-computed embedding similarities
    ├── candidates_lite.pkl        # Lightweight candidate data for reasoning
    └── best_weights.json          # Tuned scoring weights (optional)
```
