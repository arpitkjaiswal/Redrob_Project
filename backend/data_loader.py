"""
src/data_loader.py
Stream-load the candidates.jsonl file with validation and error recovery.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

REQUIRED_TOP_KEYS = {
    "candidate_id", "profile", "career_history",
    "education", "skills", "redrob_signals",
}

REQUIRED_PROFILE_KEYS = {
    "anonymized_name", "headline", "summary", "location", "country",
    "years_of_experience", "current_title", "current_company",
    "current_company_size", "current_industry",
}

COMPANY_SIZE_ENUM = {
    "1-10", "11-50", "51-200", "201-500",
    "501-1000", "1001-5000", "5001-10000", "10001+",
}


def _validate_candidate(record: dict, line_no: int) -> Optional[str]:
    """
    Returns an error string if the record is fatally malformed, else None.
    Soft issues (missing optional fields) are silently handled downstream.
    """
    missing = REQUIRED_TOP_KEYS - record.keys()
    if missing:
        return f"Line {line_no}: missing top-level keys {missing}"

    profile = record.get("profile", {})
    if not isinstance(profile, dict):
        return f"Line {line_no}: 'profile' must be a dict"

    cid = record.get("candidate_id", "")
    if not isinstance(cid, str) or not cid.startswith("CAND_"):
        return f"Line {line_no}: bad candidate_id '{cid}'"

    return None


def stream_candidates(
    path: str | Path,
    max_errors: int = 500,
) -> Generator[dict, None, None]:
    """
    Yield validated candidate records one at a time from a JSONL file.
    Tolerates up to `max_errors` bad lines before aborting.

    Args:
        path: Path to candidates.jsonl
        max_errors: Maximum number of parse/validation errors before abort
    Yields:
        dict: A single validated candidate record
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    errors = 0
    total = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors += 1
                logger.warning("Line %d: JSON parse error — %s", line_no, exc)
                if errors >= max_errors:
                    raise RuntimeError(
                        f"Exceeded max_errors={max_errors} at line {line_no}"
                    )
                continue

            # Validate structure
            err = _validate_candidate(record, line_no)
            if err:
                errors += 1
                logger.warning(err)
                if errors >= max_errors:
                    raise RuntimeError(
                        f"Exceeded max_errors={max_errors} at line {line_no}"
                    )
                continue

            total += 1
            yield record

    logger.info(
        "Loaded %d candidates from '%s' (%d lines skipped due to errors)",
        total, path, errors,
    )


def load_all_candidates(path: str | Path) -> list[dict]:
    """Load the entire JSONL file into memory. Use only when RAM allows."""
    return list(stream_candidates(path))
