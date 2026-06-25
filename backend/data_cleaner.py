"""
src/data_cleaner.py
Text normalization, date parsing, and anomaly / honeypot detection.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Optional


# ─── Text helpers ──────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lower-case, strip diacritics, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    """Preserve case but collapse whitespace and remove control chars."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Date helpers ──────────────────────────────────────────────────────────

_DATE_FORMATS = ["%Y-%m-%d", "%Y-%m", "%Y"]


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    return None


def date_diff_months(start: Optional[date], end: Optional[date]) -> int:
    """Return month difference; 0 if dates are None or inverted."""
    if start is None or end is None:
        return 0
    delta = (end.year - start.year) * 12 + (end.month - start.month)
    return max(0, delta)


# ─── Honeypot / anomaly detection ─────────────────────────────────────────

def detect_honeypot(candidate: dict) -> dict[str, bool]:
    """
    Return a dict of honeypot flags.

    Checks:
    - expert_zero_duration : claimed 'expert' with 0 months usage
    - time_travel          : stated duration > possible calendar span
    - title_desc_mismatch  : current title has no overlap with any role desc
    - skill_stuffing       : >40 skills listed
    - impossible_yoe       : years_of_experience > career_span_years
    """
    flags: dict[str, bool] = {
        "expert_zero_duration": False,
        "time_travel": False,
        "title_desc_mismatch": False,
        "skill_stuffing": False,
        "impossible_yoe": False,
    }

    skills: list[dict] = candidate.get("skills", []) or []
    profile: dict = candidate.get("profile", {}) or {}
    career: list[dict] = candidate.get("career_history", []) or []

    # Expert with 0 months
    for sk in skills:
        if (
            sk.get("proficiency") == "expert"
            and sk.get("duration_months", 0) == 0
        ):
            flags["expert_zero_duration"] = True
            break

    # Time-travel: stated duration > possible calendar span
    today = date.today()
    for job in career:
        start = parse_date(job.get("start_date"))
        end_raw = job.get("end_date")
        end = today if (job.get("is_current") or not end_raw) else parse_date(end_raw)
        calendar_months = date_diff_months(start, end)
        stated_months = int(job.get("duration_months", 0) or 0)
        if stated_months > calendar_months + 3:   # 3-month grace period
            flags["time_travel"] = True
            break

    # Skill stuffing
    if len(skills) > 40:
        flags["skill_stuffing"] = True

    # Title–description mismatch (current role desc contains no title words)
    current_title = normalize_text(profile.get("current_title", ""))
    title_words = set(w for w in current_title.split() if len(w) > 3)
    if title_words and career:
        current_jobs = [j for j in career if j.get("is_current")]
        if current_jobs:
            desc = normalize_text(current_jobs[0].get("description", ""))
            if title_words and not any(w in desc for w in title_words):
                flags["title_desc_mismatch"] = True

    # Impossible YOE
    yoe = float(profile.get("years_of_experience", 0) or 0)
    if career:
        earliest_start = min(
            (parse_date(j.get("start_date")) for j in career
             if parse_date(j.get("start_date"))),
            default=None,
        )
        if earliest_start:
            max_possible_yoe = (today - earliest_start).days / 365.25
            if yoe > max_possible_yoe + 1:
                flags["impossible_yoe"] = True

    return flags


def is_disqualified(honeypot_flags: dict[str, bool]) -> bool:
    """Hard disqualify if 2+ honeypot signals are tripped."""
    return sum(honeypot_flags.values()) >= 2


def clean_candidate(candidate: dict) -> dict:
    """
    Return a cleaned copy of the candidate record.
    Normalizes text fields and adds a 'honeypot_flags' key.
    """
    c = dict(candidate)

    # Clean profile text fields
    profile = dict(c.get("profile", {}) or {})
    for field in ("headline", "summary", "current_title", "current_company"):
        if field in profile:
            profile[field] = clean_text(profile[field])
    c["profile"] = profile

    # Clean career descriptions
    career = []
    for job in (c.get("career_history") or []):
        job = dict(job)
        job["description"] = clean_text(job.get("description", "") or "")
        job["title"] = clean_text(job.get("title", "") or "")
        career.append(job)
    c["career_history"] = career

    # Attach honeypot flags
    c["honeypot_flags"] = detect_honeypot(c)
    c["is_disqualified"] = is_disqualified(c["honeypot_flags"])

    return c
