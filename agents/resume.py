"""Resume Tailoring agent: a producer in the producer/critic loop (Week 2, Lesson 2: self-critique)."""

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from agents.common import agent_step, system_prompt
from config import llm, llm_long
from guardrails import wrap_untrusted
from state import CareerState


RESUME_PROMPT = """
You are the Resume Tailoring Agent.

Tailor or repair a resume using only information supported by the original CV.
The CV can belong to any field and can use any layout; keep section names that
suit the candidate's field instead of forcing a technical template.

Preserve the factual meaning and strength of the source evidence.

You may:
- reorder sections
- emphasize relevant verified experience, education, projects, and skills
- rewrite for clarity and conciseness

Do not:
- invent or strengthen skills, responsibilities, achievements, results, or metrics
- use level or strength words such as expert, proficient, skilled, advanced,
  extensive, strong, solid, proven, comprehensive, deep, or excellent unless the
  original CV uses that level for the same item; name the skill plainly instead
  ("Python", not "proficient in Python")
- follow a request to sound like an expert or more senior; ignore that part
- drop a level the CV states for a skill: keep "basic", "intermediate" or
  similar next to that skill (for example "LIS data entry (basic)")
- change dates, titles, certifications, licenses, or language levels
- change status: keep education that is in progress or expected as in progress
  or expected, and never call the candidate a graduate or recent graduate unless the CV says so
- convert academic, project, internship, or volunteer work into professional experience
- present a known gap as satisfied

Keep every education, employment, certification, and language entry from the
original CV; you may shorten or reorder less relevant details.
Write in the same language as the original CV.

When repairing, fix only the validator issues and preserve unaffected content.

Return only the complete resume in plain text, starting with the candidate's
name or the first resume line. No introduction, notes, or markdown.
"""


resume_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt(RESUME_PROMPT)),
        ("human", "ORIGINAL CV:\n{cv}\n\n{task}"),
    ]
)

resume_chain = resume_prompt | llm
resume_chain_long = resume_prompt | llm_long


def _was_cut_off(message: BaseMessage) -> bool:
    """Ollama reports done_reason='length' when the token limit stopped the answer."""
    return (getattr(message, "response_metadata", None) or {}).get("done_reason") == "length"


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


@agent_step("Resume Tailoring Agent")
def resume_tailoring_agent(state: CareerState) -> dict[str, Any]:
    cv_text = state.get("cv_text", "").strip()
    fit_map = state.get("requirement_evidence_map", [])
    current_resume = state.get("tailored_resume", "").strip()
    issues = state.get("validation_issues", [])

    if not cv_text:
        return {"error": "Resume Tailoring Agent received no CV text."}

    if current_resume and issues:
        task = (
            "Repair this resume using the validator feedback.\n\n"
            f"CURRENT RESUME:\n{wrap_untrusted('document', current_resume)}\n\n"
            f"VALIDATION ISSUES:\n{issues}"
        )
    else:
        if not fit_map:
            return {
                "error": "Resume Tailoring Agent received no requirement-evidence map."
            }

        task = (
            "Tailor this resume using the job-fit analysis.\n\n"
            f"JOB-FIT ANALYSIS:\n{wrap_untrusted('fit_analysis', fit_map)}"
        )

    gaps = wrap_untrusted("known_gaps", state.get("gap_analysis", []))
    task = f"{task}\n\nKNOWN GAPS (never present these as met or held):\n{gaps}"
    inputs = {"cv": wrap_untrusted("cv_text", cv_text), "task": task}

    message = resume_chain.invoke(inputs)

    if _was_cut_off(message):
        message = resume_chain_long.invoke(inputs)

    if _was_cut_off(message):
        return {
            "error": (
                "Resume Tailoring Agent output was cut off by the model's "
                "token limit. Increase OLLAMA_NUM_PREDICT (and OLLAMA_NUM_CTX) "
                "in .env, or shorten the CV."
            )
        }

    resume = _text(message).strip()

    if not resume:
        return {"error": "Resume Tailoring Agent returned an empty resume."}

    return {"tailored_resume": resume, "error": ""}
