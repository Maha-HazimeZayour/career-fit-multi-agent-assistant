"""Writer nodes that count repair attempts (Week 4, Lesson 2: retry with the error, limited loop)."""

from typing import Callable

from agents.cover_letter import cover_letter_agent
from agents.resume import resume_tailoring_agent
from state import CareerState


def _run_with_retry(
    state: CareerState,
    agent: Callable[[CareerState], dict],
    output_key: str,
) -> dict:
    """Run an agent and count repair attempts."""
    is_repair = bool(state.get(output_key) and state.get("validation_issues"))

    result = agent(state)

    if is_repair:
        result["retry_count"] = state.get("retry_count", 0) + 1

    return result


def resume_node(state: CareerState) -> dict:
    return _run_with_retry(state, resume_tailoring_agent, "tailored_resume")


def cover_letter_node(state: CareerState) -> dict:
    return _run_with_retry(state, cover_letter_agent, "cover_letter")
