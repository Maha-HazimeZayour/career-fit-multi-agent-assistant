"""Job search with MCP function calling, AI review and the user's job choice (Week 4, Lesson 3; Week 5, Lesson 2)."""

from langgraph.types import interrupt

from agents.search_query import (
    call_search_tool,
    review_search_results,
    understand_search_request,
)
from config import MAX_JOB_CHARS, MAX_JOB_RESULTS, MAX_QUERY_CHARS
from guardrails import prepare_external_text
from state import CareerState
from tools.job_search import EMPLOYMENT_TYPES, SENIORITY_LEVELS
from tools.mcp_job_client import list_tools_via_mcp, search_jobs_via_mcp

from .formatting import job_to_text

FETCH_COUNT = 20
MAX_REJECTION_EXAMPLES = 3
SEARCH_TOOL = "search_jobs"

STAGE_LABELS = {
    "entry_level": "entry level",
    "mid_level": "mid level",
    "senior": "senior",
}
SENIORITY_FOR_STAGE = {
    "entry_level": "Entry-level",
    "mid_level": "Mid-level",
    "senior": "Senior",
}
TYPE_FOR_EMPLOYMENT = {
    "internship": "Intern",
    "full_time": "Full Time",
    "part_time": "Part Time",
    "contract": "Contractor",
}


def search_arguments(intent: dict) -> tuple[dict, str]:
    """Return the search_jobs arguments and whether the agent or the fallback chose them."""
    default = {
        "query": intent.get("query", ""),
        "seniority": SENIORITY_FOR_STAGE.get(intent.get("career_stage", ""), ""),
        "employment_type": TYPE_FOR_EMPLOYMENT.get(intent.get("employment_type", ""), ""),
        "country": intent.get("country") or "",
    }

    try:
        tools = [tool for tool in list_tools_via_mcp() if tool["name"] == SEARCH_TOOL]
    except Exception:
        return default, "fallback"

    result = call_search_tool({"job_search_intent": intent, "mcp_tools": tools})
    call = result.get("tool_call")

    if result.get("error") or not call:
        return default, "fallback"

    args = call["args"]
    query = str(args.get("query") or "").strip(" \"'")[:MAX_QUERY_CHARS]
    seniority = args.get("seniority")
    employment_type = args.get("employment_type")
    country = str(args.get("country") or "").strip()

    return {
        "query": query or default["query"],
        "seniority": seniority if seniority in SENIORITY_LEVELS else default["seniority"],
        "employment_type": (
            employment_type if employment_type in EMPLOYMENT_TYPES else default["employment_type"]
        ),
        "country": country or default["country"],
    }, "agent"


def describe_preferences(intent: dict) -> str:
    parts = []

    stage = STAGE_LABELS.get(intent.get("career_stage", ""))

    if stage:
        parts.append(stage)

    if intent.get("employment_type", "any") != "any":
        parts.append(f"{intent['employment_type'].replace('_', ' ')} preferred")

    if intent.get("country"):
        parts.append(f"country: {intent['country']}")

    return " | ".join(parts)


def select_reviewed(jobs: list[dict], verdicts: dict[int, dict]) -> tuple[list[dict], list[str]]:
    """Keep the listings the AI says fit, preferred ones first."""
    fitting = []
    rejected = []

    for number, job in enumerate(jobs, 1):
        verdict = verdicts.get(number)

        if verdict and verdict.get("fits"):
            fitting.append((number, job))
        elif verdict:
            rejected.append(f"{job.get('title', '')}: {verdict.get('reason', '')}".strip(": "))

    fitting.sort(key=lambda item: not verdicts[item[0]].get("preferred", False))

    return [job for _, job in fitting[:MAX_JOB_RESULTS]], rejected[:MAX_REJECTION_EXAMPLES]


def format_job_results(
    jobs: list[dict],
    query: str,
    personalized: bool,
    preferences: str = "",
) -> str:
    lines = [
        "LIVE JOB SEARCH RESULTS",
        "Remote jobs from Himalayas (himalayas.app). Location shows where the employer hires from.",
        "",
    ]

    if personalized:
        lines += [f"Search based on CV: {query}"]

    if preferences:
        lines += [f"Preferences applied: {preferences}"]

    if personalized or preferences:
        lines += [""]

    for number, job in enumerate(jobs, 1):
        lines.append(
            f"{number}. {job.get('title', '')}\n"
            f"Company: {job.get('company', '')}\n"
            f"Location: {job.get('location', '')}\n"
            f"Level: {job.get('level', '')}\n"
            f"URL: {job.get('url', '')}\n"
        )

    return "\n".join(lines)


def no_results_message(query: str, preferences: str, found: int, rejected: list[str]) -> str:
    if not found:
        return (
            f'No matching jobs were found for "{query}" on Himalayas, which lists '
            "remote jobs only. Try a shorter or more general job title, or "
            "search without a country."
        )

    wanted = f"your preferences ({preferences}) for " if preferences else ""
    message = (
        f'No listings matched {wanted}"{query}". {found} listings were reviewed '
        "and none fit. Try a keyword search with a different title, for example "
        '"Find junior data analyst jobs".'
    )

    if rejected:
        message += "\n\nExamples:\n" + "\n".join(f"- {reason}" for reason in rejected)

    return message


def get_intent(state: CareerState) -> dict:
    query_from_cv = state.get("job_search_query", "").strip()
    intent = state.get("job_search_intent") or (
        {"query": query_from_cv} if query_from_cv else {}
    )

    if intent:
        return intent

    understood = understand_search_request(state)

    if understood.get("error"):
        raise RuntimeError(understood["error"])

    return understood["job_search_intent"]


def job_search_node(state: CareerState) -> dict:
    try:
        intent = get_intent(state)
        arguments, caller = search_arguments(intent)
        jobs = search_jobs_via_mcp(**arguments, count=FETCH_COUNT)

        found = len(jobs)
        rejected = []
        note = ""

        if jobs:
            reviewed = review_search_results(
                {"job_search_intent": intent, "job_search_results": jobs}
            )

            if reviewed.get("error"):
                jobs = jobs[:MAX_JOB_RESULTS]
                note = (
                    "NOTE: your preferences could not be checked automatically, "
                    f"so these results are not filtered. ({reviewed['error']})"
                )
            else:
                jobs, rejected = select_reviewed(jobs, reviewed["verdicts"])

        query = arguments["query"]
        preferences = describe_preferences(intent)

        if jobs:
            personalized = bool(state.get("job_search_query", "").strip())
            response = format_job_results(jobs, query, personalized, preferences)
            response += f"\n{note}" if note else ""
        else:
            response = no_results_message(query, preferences, found, rejected)

        return {
            "job_search_results": jobs,
            "job_search_intent": intent,
            "search_tool_caller": caller,
            "final_response": response,
            "error": "",
        }

    except Exception as error:
        return {
            "job_search_results": [],
            "final_response": f"Job search failed.\n\n{error}",
            "error": "",
        }


def parse_job_choice(choice: object, job_count: int) -> int | None:
    """Turn the candidate's answer ("2") into a list index, or None."""
    text = str(choice or "").strip()

    if not text.isdigit():
        return None

    index = int(text) - 1

    return index if 0 <= index < job_count else None


def select_job_node(state: CareerState) -> dict:
    """Human-in-the-loop step: pause until the candidate picks a job."""
    jobs = state.get("job_search_results", [])

    choice = interrupt(
        {
            "type": "select_job",
            "message": state.get("final_response", ""),
            "job_count": len(jobs),
        }
    )

    index = parse_job_choice(choice, len(jobs))

    if index is None:
        return {"selected_job": {}}

    job = jobs[index]
    job_text, notes = prepare_external_text(
        job_to_text(job), MAX_JOB_CHARS, "job description"
    )

    return {
        "selected_job": job,
        "job_text": job_text,
        "input_warnings": [*state.get("input_warnings", []), *notes],
        "error": "",
    }
