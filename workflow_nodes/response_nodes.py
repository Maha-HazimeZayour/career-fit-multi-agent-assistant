"""Final response nodes, including clear failure messages (Week 4, Lesson 2: failure recovery)."""

from state import CareerState
from workflow_routes import COVER_LETTER_ROUTES

from .formatting import (
    format_evidence_item,
    format_gap_detailed,
    format_gap_short,
    with_notes,
)


def fit_response_node(state: CareerState) -> dict:
    lines = ["CAREER FIT ANALYSIS", ""]
    lines.extend(
        format_evidence_item(item)
        for item in state.get("requirement_evidence_map", [])
    )

    gaps = state.get("gap_analysis", [])

    if gaps:
        lines.extend(["", "MAIN GAPS", ""])
        lines.extend(format_gap_detailed(gap) for gap in gaps)

    return {
        "final_response": with_notes(state, "\n".join(lines)),
        "error": "",
    }


def validated_response_node(state: CareerState) -> dict:
    route = state.get("route")

    if route in COVER_LETTER_ROUTES:
        response = state.get("cover_letter", "")

    elif route == "full_flow":
        gaps = state.get("gap_analysis", [])
        gap_text = (
            "\n".join(format_gap_short(gap) for gap in gaps)
            if gaps
            else "No major gaps were identified."
        )
        response = (
            "CAREER FIT & GAP SUMMARY\n\n"
            f"{gap_text}\n\n"
            "TAILORED RESUME\n\n"
            f"{state.get('tailored_resume', '')}"
        )

    else:
        response = state.get("tailored_resume", "")

    return {
        "final_response": with_notes(state, response),
        "error": "",
    }


def failed_validation_node(state: CareerState) -> dict:
    issues = state.get("validation_issues", [])
    issue_text = (
        "\n".join(f"- {issue}" for issue in issues)
        or "No additional validator details were returned."
    )

    return {
        "final_response": (
            "The generated application document could not pass the "
            "automatic quality and truthfulness checks after the allowed "
            "repair attempts.\n\n"
            f"Remaining issues:\n{issue_text}"
        ),
        "error": "",
    }


def workflow_error_node(state: CareerState) -> dict:
    error = state.get("error", "Unknown error.")

    return {
        "final_response": (
            "The workflow stopped because a processing step failed.\n\n"
            f"{error}"
        ),
        "error": error,
    }


def unknown_request_node(state: CareerState) -> dict:
    return {
        "final_response": (
            "The request could not be matched to a supported workflow. "
            "Ask for fit analysis, resume tailoring, a cover letter, "
            "job search, or CV-based job search."
        ),
        "error": "",
    }
