"""Supervisor agent: LLM-based routing to one workflow (Week 4, Lesson 2)."""

from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from agents.common import agent_step, system_prompt
from config import llm
from guardrails import wrap_untrusted
from state import CareerState


SUPERVISOR_PROMPT = """
You are the Supervisor Agent.

GOAL
Route the user's request to the smallest workflow that fully satisfies it.

AVAILABLE ROUTES
- fit_analysis: candidate-job fit or gap analysis
- resume_tailoring: tailor a resume for a specific job
- cover_letter: write a cover letter for a specific job
- full_flow: fit analysis plus resume tailoring
- job_search: search jobs using a keyword, title, or skill
- personalized_job_search: search jobs based on the candidate's CV
- job_search_cover_letter: find jobs from the CV, then prepare a cover letter
- unknown: request does not match a supported workflow

Do not perform the task.
Only choose the route.

Return only a valid JSON object with one field:
route

The route value must be exactly one of the available routes.
"""


class RouteDecision(BaseModel):
    route: Literal[
        "fit_analysis",
        "resume_tailoring",
        "cover_letter",
        "full_flow",
        "job_search",
        "personalized_job_search",
        "job_search_cover_letter",
        "unknown",
    ]


supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt(SUPERVISOR_PROMPT)),
        ("human", "{request}"),
    ]
)

supervisor_chain = supervisor_prompt | llm.bind(format="json")


@agent_step("Supervisor Agent", on_error={"route": "unknown"})
def supervisor_agent(state: CareerState) -> dict[str, Any]:
    user_request = state.get("user_request", "").strip()

    if not user_request:
        return {
            "route": "unknown",
            "error": "Supervisor Agent received no user request.",
        }

    response = supervisor_chain.invoke(
        {
            "request": wrap_untrusted(
                "user_request",
                user_request,
            )
        }
    )

    try:
        result = RouteDecision.model_validate_json(response.content)

    except ValidationError as error:
        return {
            "route": "unknown",
            "error": (
                "Supervisor Agent returned invalid structured output: "
                f"{error}"
            ),
        }

    return {
        "route": result.route,
        "error": "",
    }
