"""Text helpers for the final responses."""

from state import CareerState


def format_gap_detailed(gap: dict) -> str:
    return (
        f"- {gap.get('requirement', '')}\n"
        f"  Gap type: {gap.get('gap_type', '')}\n"
        f"  {gap.get('explanation', '')}"
    )


def format_gap_short(gap: dict) -> str:
    return f"- {gap.get('requirement', '')}: {gap.get('explanation', '')}"


def format_evidence_item(item: dict) -> str:
    return (
        f"- {item.get('requirement', '')}\n"
        f"  Match: {item.get('match_level', '')}\n"
        f"  {item.get('explanation', '')}"
    )


def job_to_text(job: dict) -> str:
    """Turn a job-search result into the text the Job Requirements Agent reads."""
    return (
        f"Job Title: {job.get('title', '')}\n"
        f"Company: {job.get('company', '')}\n"
        f"Location: {job.get('location', '')}\n"
        f"Level: {job.get('level', '')}\n\n"
        f"Job Description:\n{job.get('description', '')}"
    )


def with_notes(state: CareerState, text: str) -> str:
    """Append input-guard warnings so the user knows what was neutralized."""
    warnings = state.get("input_warnings", [])

    if not warnings:
        return text

    notes = "\n".join(f"- {warning}" for warning in warnings)

    return f"{text}\n\nNOTES\n\n{notes}"
