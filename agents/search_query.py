"""Job Search Query agent: search intent, MCP function calling and result review (Week 2, Lesson 1; Week 4, Lesson 3)."""

from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.common import agent_step, structured, system_prompt
from config import FIT_MAX_CONCURRENCY, MAX_QUERY_CHARS, llm
from guardrails import wrap_untrusted
from state import CareerState


SEARCH_QUERY_PROMPT = """
You are the Job Search Query Agent.

Turn the user's request and, when provided, the candidate's verified CV
evidence into one structured job-search intent. You do the understanding;
later steps only pass your fields on, so decide them from the meaning of
the text, in any wording and any language.

FIELDS
query
  One concise, common job title or short search phrase for a job-search API.
  With CV evidence, base it on the candidate's demonstrated education,
  skills, and experience and choose the most realistic role (the strongest
  overall match if several areas are supported). Without CV evidence, use the
  role or topic the user asked for, without the surrounding words.

career_stage
  entry_level, mid_level, senior, or unspecified.
  With CV evidence, judge it from the evidence: studies, internships, and
  junior roles mean entry_level. The user's own words can also show it, for
  example someone starting out, looking for a first role, or wanting an
  internship. Never choose a stage higher than the evidence supports, even if
  the user asks for a more senior role. Without CV evidence, use only what the
  user says, otherwise unspecified.

employment_type
  internship, full_time, part_time, contract, or any.
  Use a specific value only when the user states a preference for it.
  If the user accepts several types, use any.

country
  The country the user asks to work in, as its English name (for example
  "Germany" or "United Arab Emirates"); for a city, its country. Use null when
  the user names no country or only a region such as Europe. Take it only from
  the user's request, never from the CV.

TRUTHFULNESS
Do not add qualifications, roles, or specialties the evidence does not
support, and do not increase the candidate's demonstrated seniority.
"""

REVIEW_PROMPT = """
You are the Job Search Query Agent, reviewing live search results.

For every listing decide whether it fits the search intent. Judge by meaning,
using only the fields shown for each listing.

fits (career stage and topic)
  topic: the listing's title must be the target role or a closely related
    role. Reject a listing for an unrelated field (for example accounting when
    the target role is customer success).
  career stage: the listing's level and title must suit the intended stage.
    entry_level means junior, entry-level, intern, trainee, or graduate roles;
    a listing open to any level also fits. Senior, lead, manager, and
    mid-level roles do not fit an entry-level candidate. mid_level and senior
    work the same way for their own stage. unspecified accepts any stage.

preferred
  true only when the listing also matches the employment type the user
  prefers (for example an internship). Otherwise false.

Return one verdict for every listing, using the listing's number, with a
short reason. Do not invent details that are not shown.
"""

TOOL_CALL_PROMPT = """
You are the Job Search Query Agent. You have been given a job-search tool.

Call the tool once to find jobs for the search intent below:
- query: the target role
- seniority: Entry-level for entry_level, Mid-level for mid_level, Senior for
  senior, and empty for unspecified
- employment_type: Intern for internship, Full Time for full_time, Part Time
  for part_time, Contractor for contract, and empty for any
- country: the country in the intent, or empty when it is "none"
Do not answer in text; call the tool.
"""

NO_CV = "Not provided (keyword search): use only the user's request."


class JobSearchIntent(BaseModel):
    query: str
    career_stage: Literal["entry_level", "mid_level", "senior", "unspecified"] = "unspecified"
    employment_type: Literal["internship", "full_time", "part_time", "contract", "any"] = "any"
    country: str | None = None


class JobVerdict(BaseModel):
    number: int
    fits: bool
    preferred: bool = False
    reason: str = ""


class JobReview(BaseModel):
    verdicts: list[JobVerdict] = Field(default_factory=list)


search_query_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt(SEARCH_QUERY_PROMPT)),
            (
                "human",
                "USER REQUEST:\n{request}\n\n"
                "CANDIDATE PROFILE:\n{profile}\n\n"
                "VERIFIED EVIDENCE:\n{evidence}",
            ),
        ]
    )
    | structured(JobSearchIntent)
)

review_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt(REVIEW_PROMPT)),
            (
                "human",
                "SEARCH INTENT:\n{intent}\n\nLISTINGS:\n{listings}",
            ),
        ]
    )
    | structured(JobReview)
)

tool_call_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt(TOOL_CALL_PROMPT)),
        ("human", "SEARCH INTENT:\n{intent}"),
    ]
)

tool_model = llm

REVIEW_BATCH_SIZE = 10


def _clean_intent(result: JobSearchIntent) -> dict[str, Any]:
    """Normalize the model's answer; the intent goes to an external API."""
    return {
        "query": result.query.strip().strip("\"'")[:MAX_QUERY_CHARS].strip(),
        "career_stage": result.career_stage,
        "employment_type": result.employment_type,
        "country": (result.country or "").strip() or None,
    }


def intent_text(intent: dict) -> str:
    """The intent as plain lines for the review prompt (formatting only)."""
    return (
        f"target role: {intent.get('query') or 'not stated'}\n"
        f"career stage: {intent.get('career_stage', 'unspecified')}\n"
        f"preferred employment type: {intent.get('employment_type', 'any')}\n"
        f"country: {intent.get('country') or 'none'}"
    )


def _listing_line(number: int, job: dict) -> str:
    return (
        f"{number}. {job.get('title', '')} | "
        f"level: {job.get('level') or 'not stated'} | "
        f"type: {job.get('job_type') or 'not stated'}"
    )


@agent_step("Job Search Query Agent")
def job_search_query_agent(state: CareerState) -> dict[str, Any]:
    """Graph node for CV-based search: request and CV evidence become a search intent."""
    profile = state.get("candidate_profile", {})
    evidence = state.get("candidate_evidence", [])

    if not evidence:
        return {"error": "Job Search Query Agent received no candidate evidence."}

    result = search_query_chain.invoke(
        {
            "request": wrap_untrusted("user_request", state.get("user_request", "")),
            "profile": wrap_untrusted("candidate_profile", profile),
            "evidence": wrap_untrusted("candidate_evidence", evidence),
        }
    )

    intent = _clean_intent(result)

    if not intent["query"]:
        return {"error": "Job Search Query Agent returned an empty query."}

    return {
        "job_search_query": intent["query"],
        "job_search_intent": intent,
        "error": "",
    }


@agent_step("Job Search Query Agent")
def understand_search_request(state: CareerState) -> dict[str, Any]:
    """Keyword search: the same intent, understood from the request alone."""
    request = state.get("user_request", "").strip()

    if not request:
        return {"error": "Job Search Query Agent received no request."}

    result = search_query_chain.invoke(
        {
            "request": wrap_untrusted("user_request", request),
            "profile": NO_CV,
            "evidence": NO_CV,
        }
    )

    intent = _clean_intent(result)

    if not intent["query"]:
        return {"error": "Job Search Query Agent returned an empty query."}

    return {"job_search_intent": intent, "error": ""}


def _as_function(tool: dict) -> dict:
    """An MCP tool description in the function-calling format Ollama expects."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


@agent_step("Job Search Query Agent (tool call)")
def call_search_tool(state: CareerState) -> dict[str, Any]:
    """Function calling: the agent receives the MCP tools and makes the tool call."""
    tools = state.get("mcp_tools") or []
    intent = state.get("job_search_intent") or {}

    if not tools:
        return {"tool_call": None, "error": ""}

    chain = tool_call_prompt | tool_model.bind_tools([_as_function(t) for t in tools])
    message = chain.invoke(
        {"intent": wrap_untrusted("search_intent", intent_text(intent))}
    )

    names = {tool["name"] for tool in tools}

    for call in getattr(message, "tool_calls", None) or []:
        if call.get("name") in names:
            return {
                "tool_call": {"name": call["name"], "args": dict(call.get("args") or {})},
                "error": "",
            }

    return {"tool_call": None, "error": ""}


@agent_step("Job Search Query Agent (result review)")
def review_search_results(state: CareerState) -> dict[str, Any]:
    """Ask the AI whether each returned listing fits the intent."""
    intent = state.get("job_search_intent") or {}
    jobs = state.get("job_search_results") or []

    numbered = list(enumerate(jobs, 1))
    batches = [
        numbered[start:start + REVIEW_BATCH_SIZE]
        for start in range(0, len(numbered), REVIEW_BATCH_SIZE)
    ]

    reviews = review_chain.batch(
        [
            {
                "intent": wrap_untrusted("search_intent", intent_text(intent)),
                "listings": wrap_untrusted(
                    "job_listings",
                    "\n".join(_listing_line(number, job) for number, job in batch),
                ),
            }
            for batch in batches
        ],
        config={"max_concurrency": FIT_MAX_CONCURRENCY},
    )

    verdicts: dict[int, dict] = {}

    for batch, review in zip(batches, reviews):
        valid = {number for number, _ in batch}

        for verdict in review.verdicts:
            if verdict.number in valid and verdict.number not in verdicts:
                verdicts[verdict.number] = verdict.model_dump()

    return {"verdicts": verdicts, "error": ""}
