"""Shared state that every agent reads and writes (Week 4, Lesson 2: shared context; Week 3, Lesson 1: short-term memory)."""

from typing import Any, TypedDict


class EvidenceItem(TypedDict, total=False):
    evidence: str
    evidence_type: str
    source: str
    stated_level: str | None
    period: str | None


class RequirementItem(TypedDict, total=False):
    requirement: str
    requirement_type: str
    priority: str


class EvidenceMatch(TypedDict, total=False):
    requirement: str
    priority: str
    match_level: str
    evidence: str
    evidence_type: str
    gap_type: str
    explanation: str


class CareerState(TypedDict, total=False):
    user_request: str
    route: str

    cv_text: str
    job_text: str

    input_warnings: list[str]
    request_screened: bool

    candidate_profile: dict[str, Any]
    candidate_evidence: list[EvidenceItem]

    job_profile: dict[str, Any]
    job_requirements: list[RequirementItem]

    requirement_evidence_map: list[EvidenceMatch]
    gap_analysis: list[dict[str, Any]]

    job_search_query: str
    job_search_intent: dict[str, Any]
    job_search_results: list[dict[str, Any]]
    selected_job: dict[str, Any]
    search_tool_caller: str

    tailored_resume: str

    cover_letter: str

    validation_status: str
    validation_issues: list[str]
    retry_count: int
    max_retries: int

    final_response: str
    error: str
