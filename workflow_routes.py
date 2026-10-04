"""Conditional edges: routing by request and the limit on repair attempts (Week 4, Lesson 2)."""

from typing import Literal

from config import MAX_RETRIES
from state import CareerState

CANDIDATE_ROUTES = {
    "fit_analysis",
    "resume_tailoring",
    "cover_letter",
    "full_flow",
    "personalized_job_search",
    "job_search_cover_letter",
}

SEARCH_FIRST_ROUTES = {
    "personalized_job_search",
    "job_search_cover_letter",
}

COVER_LETTER_ROUTES = {
    "cover_letter",
    "job_search_cover_letter",
}


def route_from_state(
    state: CareerState,
) -> Literal["candidate", "job_search", "unknown", "error"]:
    if state.get("error"):
        return "error"

    route = state.get("route", "unknown")

    if route == "job_search":
        return "job_search"

    if route in CANDIDATE_ROUTES:
        return "candidate"

    return "unknown"


def route_after_candidate(
    state: CareerState,
) -> Literal["job", "job_search_query", "error"]:
    if state.get("error"):
        return "error"

    if state.get("route") in SEARCH_FIRST_ROUTES:
        return "job_search_query"

    return "job"


def route_after_agent(
    state: CareerState,
) -> Literal["continue", "error"]:
    return "error" if state.get("error") else "continue"


def route_after_job_search(
    state: CareerState,
) -> Literal["select_job", "end"]:
    """Only the search-to-cover-letter workflow continues after the search."""
    if (
        state.get("route") == "job_search_cover_letter"
        and state.get("job_search_results")
    ):
        return "select_job"

    return "end"


def route_after_selection(
    state: CareerState,
) -> Literal["job", "end"]:
    return "job" if state.get("selected_job") else "end"


def route_after_fit(
    state: CareerState,
) -> Literal["fit_response", "resume", "cover_letter", "error"]:
    if state.get("error"):
        return "error"

    route = state.get("route")

    if route == "fit_analysis":
        return "fit_response"

    if route in COVER_LETTER_ROUTES:
        return "cover_letter"

    return "resume"


def route_to_validator(
    state: CareerState,
) -> Literal["validator", "error"]:
    return "error" if state.get("error") else "validator"


def route_after_validation(
    state: CareerState,
) -> Literal[
    "validated",
    "resume",
    "cover_letter",
    "failed_validation",
    "error",
]:
    if state.get("error"):
        return "error"

    if state.get("validation_status") == "pass":
        return "validated"

    if state.get("retry_count", 0) >= state.get("max_retries", MAX_RETRIES):
        return "failed_validation"

    if state.get("route") in COVER_LETTER_ROUTES:
        return "cover_letter"

    return "resume"
