"""
Task 3: Database Schema Design — Python Data Models

Defines structured dataclasses for all candidate profile components.
These models enforce type safety and provide a clean interface for
the feature extraction and ranking pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Skill:
    name: str
    proficiency: str  # beginner, intermediate, advanced, expert
    endorsements: int = 0
    duration_months: int = 0


@dataclass
class CareerEntry:
    company: str
    title: str
    start_date: str
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = "unknown"


@dataclass
class Certification:
    name: str
    issuer: str
    year: int


@dataclass
class Language:
    language: str
    proficiency: str  # basic, conversational, professional, native


@dataclass
class SalaryRange:
    min: float
    max: float


@dataclass
class RedrobSignals:
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict  # skill_name -> score 0-100
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str  # remote, hybrid, onsite, flexible
    willing_to_relocate: bool
    github_activity_score: float  # -1 if no GitHub
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float  # -1 if no prior offers
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass
class CandidateProfile:
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


@dataclass
class Candidate:
    """Full candidate record with all sub-components."""
    candidate_id: str
    profile: CandidateProfile
    career_history: list[CareerEntry]
    education: list[Education]
    skills: list[Skill]
    certifications: list[Certification] = field(default_factory=list)
    languages: list[Language] = field(default_factory=list)
    redrob_signals: RedrobSignals = None

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        """Parse a candidate from a raw JSON dict."""
        profile_data = data["profile"]
        profile = CandidateProfile(
            anonymized_name=profile_data.get("anonymized_name", ""),
            headline=profile_data.get("headline", ""),
            summary=profile_data.get("summary", ""),
            location=profile_data.get("location", ""),
            country=profile_data.get("country", ""),
            years_of_experience=float(profile_data.get("years_of_experience", 0)),
            current_title=profile_data.get("current_title", ""),
            current_company=profile_data.get("current_company", ""),
            current_company_size=profile_data.get("current_company_size", ""),
            current_industry=profile_data.get("current_industry", ""),
        )

        career_history = [
            CareerEntry(
                company=c.get("company", ""),
                title=c.get("title", ""),
                start_date=c.get("start_date", ""),
                end_date=c.get("end_date"),
                duration_months=int(c.get("duration_months", 0)),
                is_current=bool(c.get("is_current", False)),
                industry=c.get("industry", ""),
                company_size=c.get("company_size", ""),
                description=c.get("description", ""),
            )
            for c in data.get("career_history", [])
        ]

        education = [
            Education(
                institution=e.get("institution", ""),
                degree=e.get("degree", ""),
                field_of_study=e.get("field_of_study", ""),
                start_year=int(e.get("start_year", 0)),
                end_year=int(e.get("end_year", 0)),
                grade=e.get("grade"),
                tier=e.get("tier", "unknown"),
            )
            for e in data.get("education", [])
        ]

        skills = [
            Skill(
                name=s.get("name", ""),
                proficiency=s.get("proficiency", "beginner"),
                endorsements=int(s.get("endorsements", 0)),
                duration_months=int(s.get("duration_months", 0)),
            )
            for s in data.get("skills", [])
        ]

        certifications = [
            Certification(
                name=c.get("name", ""),
                issuer=c.get("issuer", ""),
                year=int(c.get("year", 0)),
            )
            for c in data.get("certifications", [])
        ]

        languages = [
            Language(
                language=l.get("language", ""),
                proficiency=l.get("proficiency", "basic"),
            )
            for l in data.get("languages", [])
        ]

        signals_data = data.get("redrob_signals", {})
        salary_data = signals_data.get("expected_salary_range_inr_lpa", {})
        signals = RedrobSignals(
            profile_completeness_score=float(signals_data.get("profile_completeness_score", 0)),
            signup_date=signals_data.get("signup_date", ""),
            last_active_date=signals_data.get("last_active_date", ""),
            open_to_work_flag=bool(signals_data.get("open_to_work_flag", False)),
            profile_views_received_30d=int(signals_data.get("profile_views_received_30d", 0)),
            applications_submitted_30d=int(signals_data.get("applications_submitted_30d", 0)),
            recruiter_response_rate=float(signals_data.get("recruiter_response_rate", 0)),
            avg_response_time_hours=float(signals_data.get("avg_response_time_hours", 0)),
            skill_assessment_scores=signals_data.get("skill_assessment_scores", {}),
            connection_count=int(signals_data.get("connection_count", 0)),
            endorsements_received=int(signals_data.get("endorsements_received", 0)),
            notice_period_days=int(signals_data.get("notice_period_days", 0)),
            expected_salary_range_inr_lpa=SalaryRange(
                min=float(salary_data.get("min", 0)),
                max=float(salary_data.get("max", 0)),
            ),
            preferred_work_mode=signals_data.get("preferred_work_mode", "flexible"),
            willing_to_relocate=bool(signals_data.get("willing_to_relocate", False)),
            github_activity_score=float(signals_data.get("github_activity_score", -1)),
            search_appearance_30d=int(signals_data.get("search_appearance_30d", 0)),
            saved_by_recruiters_30d=int(signals_data.get("saved_by_recruiters_30d", 0)),
            interview_completion_rate=float(signals_data.get("interview_completion_rate", 0)),
            offer_acceptance_rate=float(signals_data.get("offer_acceptance_rate", -1)),
            verified_email=bool(signals_data.get("verified_email", False)),
            verified_phone=bool(signals_data.get("verified_phone", False)),
            linkedin_connected=bool(signals_data.get("linkedin_connected", False)),
        )

        return cls(
            candidate_id=data["candidate_id"],
            profile=profile,
            career_history=career_history,
            education=education,
            skills=skills,
            certifications=certifications,
            languages=languages,
            redrob_signals=signals,
        )
