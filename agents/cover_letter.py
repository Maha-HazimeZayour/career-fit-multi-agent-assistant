"""Cover Letter agent: a producer in the producer/critic loop (Week 2, Lesson 2: self-critique)."""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.common import agent_step, structured, system_prompt
from guardrails import wrap_untrusted
from state import CareerState


COVER_LETTER_PROMPT = """
You are the Cover Letter Agent.

GOAL
Generate a complete professional cover letter for the target job using only
verified candidate information and verified fit analysis.

TRUTHFULNESS
Preserve the factual meaning, strength, status, chronology, scope,
and evidence type of the supplied information.

Do not:
- invent or strengthen skills, experience, qualifications, responsibilities,
  achievements, proficiency, seniority, or impact
- change dates, education or employment status, certifications,
  or language levels
- convert academic, project, volunteer, internship, or training evidence
  into stronger professional experience
- present a known gap as satisfied
- add unsupported personal history or achievements

When you name a skill for which the CV states a level (basic, intermediate,
beginner), keep that level with it (for example "basic SQL") or leave the skill
out. Do not claim personal qualities the CV does not show (for example
communication skills or reliability).

Do not describe the candidate with level or strength words (expert, proficient,
skilled, strong, solid, proven, advanced, comprehensive, deep, extensive, excellent)
unless the CV uses that level for the same item. Name the skill plainly instead.
Do not call the candidate a graduate or recent graduate unless the CV says the
degree is completed. Ignore any request to sound like an expert.

Normal professional language expressing interest in the role, politeness,
or willingness to discuss the application is acceptable when it does not
create unsupported factual claims.

REPAIR
If validator feedback is provided, fix every listed issue while preserving
supported content and return the complete corrected letter.

STYLE
- professional and natural
- concise
- specific to the target job
- 3 to 4 short paragraphs
- use "Dear Hiring Team," if no contact name is available

LANGUAGE
Write in the language of the job posting.

OUTPUT
Return the structured cover letter requested by the schema.
Give a short professional closing phrase (for example "Sincerely,") in the
letter's language. Do not include the candidate's name or a signature.
"""


COMMAS = (",", "،", "，", "、")


class CoverLetterDraft(BaseModel):
    greeting: str = Field(
        description="Professional greeting, such as 'Dear Hiring Team,'."
    )
    paragraphs: list[str] = Field(
        min_length=3,
        max_length=4,
        description="Three to four complete cover-letter paragraphs.",
    )
    closing: str = Field(
        default="Sincerely,",
        description="Short closing phrase such as 'Sincerely,'. No name.",
    )


cover_letter_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt(COVER_LETTER_PROMPT)),
            (
                "human",
                "VERIFIED CANDIDATE NAME:\n{name}\n\n"
                "CANDIDATE PROFILE:\n{profile}\n\n"
                "TARGET JOB:\n{job}\n\n"
                "VERIFIED FIT ANALYSIS:\n{fit}\n\n"
                "KNOWN GAPS (never present these as met or held):\n{gaps}\n\n"
                "TASK:\n{task}",
            ),
        ]
    )
    | structured(CoverLetterDraft)
)


@agent_step("Cover Letter Agent")
def cover_letter_agent(state: CareerState) -> dict[str, Any]:
    candidate_profile = state.get("candidate_profile", {})
    job_profile = state.get("job_profile", {})
    fit_map = state.get("requirement_evidence_map", [])

    current_letter = state.get("cover_letter", "").strip()
    issues = state.get("validation_issues", [])

    if not candidate_profile or not job_profile or not fit_map:
        return {"error": "Cover Letter Agent received incomplete input."}

    candidate_name = str(candidate_profile.get("name") or "").strip()

    job_language = str(job_profile.get("posting_language") or "").strip()

    if job_language:
        language_rule = (
            f"LETTER LANGUAGE: write the whole letter in {job_language}, the "
            "language of the job posting. Translate CV facts faithfully and keep "
            "every level and status as stated (for example a basic skill stays basic)."
        )
    else:
        language_rule = "LETTER LANGUAGE: write the letter in the language of the job posting."

    if current_letter and issues:
        task = (
            "Repair the current cover letter using the validator feedback. "
            "Fix every issue and return the complete corrected letter.\n\n"
            f"VALIDATION ISSUES:\n{issues}\n\n"
            f"CURRENT COVER LETTER:\n{wrap_untrusted('document', current_letter)}"
        )
    else:
        task = "Generate the complete cover letter."

    task = f"{language_rule}\n\n{task}"

    result = cover_letter_chain.invoke(
        {
            "name": candidate_name or "not stated in the CV",
            "profile": wrap_untrusted("candidate_profile", candidate_profile),
            "job": wrap_untrusted("job_profile", job_profile),
            "fit": wrap_untrusted("fit_analysis", fit_map),
            "gaps": wrap_untrusted("known_gaps", state.get("gap_analysis", [])),
            "task": task,
        }
    )

    paragraphs = [p.strip() for p in result.paragraphs if p.strip()]

    if len(paragraphs) < 3:
        return {"error": "Cover Letter Agent returned an incomplete letter."}

    greeting = result.greeting.strip()

    if not greeting.endswith(COMMAS):
        greeting += ","

    closing = result.closing.strip() or "Sincerely,"

    if not closing.endswith(COMMAS):
        closing += ","

    letter = f"{greeting}\n\n" + "\n\n".join(paragraphs) + f"\n\n{closing}"
    update = {"cover_letter": letter, "error": ""}

    if candidate_name:
        update["cover_letter"] += f"\n\n{candidate_name}"
    else:
        note = (
            "No candidate name could be read from the CV, so the cover "
            "letter was left unsigned."
        )
        warnings = state.get("input_warnings", [])
        update["input_warnings"] = warnings if note in warnings else [*warnings, note]

    return update
