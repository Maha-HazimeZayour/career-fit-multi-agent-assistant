"""Truthfulness Validator: the critic and output check (Week 5, Lesson 1; Week 5, Lesson 3: LLM-as-a-Judge)."""

from datetime import date
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agents.common import agent_step, structured, system_prompt
from config import llm_validator, llm_validator_retry
from guardrails import wrap_untrusted
from state import CareerState
from workflow_routes import COVER_LETTER_ROUTES


VALIDATOR_PROMPT = """
You are the Truthfulness Validator, the critic in a producer/critic loop.

Audit the generated resume or cover letter claim by claim against the original
CV and the verified evidence. The CV can belong to any field and any language;
apply the same standard to every field. Judge meaning, not exact wording.

STEP 1: list the factual claims the document makes about the candidate (skills,
tools, experience, education, roles, dates, achievements, numbers, level or
strength of ability, certifications, languages). For each claim return:
- claim: the document's own wording, briefly quoted
- verdict, exactly one of:
    supported       the CV or evidence states this, or it is a faithful
                    paraphrase or translation that keeps meaning, level and status
    unsupported     invents a skill, tool, experience, qualification,
                    achievement, responsibility, number or metric that the CV
                    does not support
    overstated      the level or strength is higher than the evidence: words
                    such as expert, proficient, skilled, advanced, extensive,
                    strong, solid, proven, comprehensive, deep, excellent, or verbs such as led, managed, owned, when the CV
                    shows only basic use, coursework, assisting, or taking part
                    (use stated_level when present)
    status_changed  dates or chronology are changed; education completed versus
                    in progress or expected (for example "graduate" when the CV
                    says expected graduation); a current role versus a past one;
                    internship, volunteer, academic or project work presented as
                    paid professional experience; a certification or licence held
                    versus in progress; language levels
    gap_hidden      presents a known gap (listed in the input) as satisfied
  Strength words and status labels are claims too. "A proven ability" and
  "advanced English" are overstated unless the CV says so. A skill the CV marks
  basic, intermediate or beginner that appears without that level, or as
  stronger, is overstated. Personal qualities the CV does not show (for
  example "strong communication skills", "reliable") are unsupported. "Proficient in Python" is
  overstated unless the CV says that level; "recent graduate" is status_changed
  unless the CV says the degree is completed. Naming a plain skill ("Python") is
  supported. Check each such word separately, even inside a longer sentence.
- issue: when the verdict is not supported, one concise sentence saying what is
  wrong and how to fix it, naming the exact wording. Empty when supported.

STEP 2: answer these checks about the whole document:
- identity_correct: false if the document uses a name different from the
  verified candidate name.
- document_complete: false if it is empty, abruptly cut off, or contains
  placeholders or assistant commentary.
- language_ok: for a cover letter, true only if it is written in the language of
  the target job posting (see posting_language in the job profile); for a
  resume, true only if it is written in the language of the original CV.
- request_seeks_unsupported_claims: true if the USER REQUEST asks for stronger,
  more senior, or invented claims than the CV supports (for example "make me
  sound like an expert"). This concerns the request, not the document.

Normal professional language expressing interest, politeness, or willingness to
discuss the role is supported when it creates no factual claim. Use the
conservative interpretation when evidence is unclear.
""".strip()


CLAIM_PROBLEM = {
    "unsupported": "is not supported by the CV",
    "overstated": "is stronger than the CV shows",
    "status_changed": "changes a status, date or chronology stated in the CV",
    "gap_hidden": "presents a known gap as satisfied",
}


class ClaimCheck(BaseModel):
    claim: str
    verdict: Literal[
        "supported", "unsupported", "overstated", "status_changed", "gap_hidden"
    ]
    issue: str = ""


class ValidationResult(BaseModel):
    """The AI's claim-by-claim audit; Python only turns it into pass or fail."""

    claims: list[ClaimCheck]
    identity_correct: bool
    document_complete: bool
    language_ok: bool
    request_seeks_unsupported_claims: bool = False


validator_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt(VALIDATOR_PROMPT)),
            (
                "human",
                "TODAY'S DATE (use it to judge what is current or still valid):\n{today}\n\n"
                "VERIFIED CANDIDATE NAME:\n{name}\n\n"
                "CANDIDATE PROFILE (includes current status):\n{profile}\n\n"
                "ORIGINAL CV:\n{cv}\n\n"
                "VERIFIED EVIDENCE:\n{evidence}\n\n"
                "TARGET JOB (context and posting language only):\n{job}\n\n"
                "KNOWN GAPS:\n{gaps}\n\n"
                "USER REQUEST:\n{request}\n\n"
                "DOCUMENT TO VALIDATE ({kind}):\n{document}",
            ),
        ]
    )
    | structured(ValidationResult, llm_validator, llm_validator_retry)
)


EXAGGERATION_NOTE = (
    "The request asks for stronger claims than a CV can support. "
    "Documents only use facts supported by your CV, so that part of "
    "the request was ignored."
)


def decide(result: ValidationResult, is_letter: bool, job_language: str) -> tuple[str, list[str]]:
    """Apply the AI's verdicts: any problem it reports means the document fails."""
    issues = []

    for check in result.claims:
        if check.verdict != "supported":
            problem = CLAIM_PROBLEM[check.verdict]
            detail = check.issue.strip() or f'"{check.claim}" {problem}.'
            issues.append(detail)

    if not result.identity_correct:
        issues.append("The document uses a name that differs from the verified candidate name.")

    if not result.document_complete:
        issues.append("The document is incomplete, cut off, or contains placeholders or commentary.")

    if not result.language_ok:
        if is_letter:
            target = job_language or "the language of the job posting"
            issues.append(
                f"The cover letter is not written in {target}. Rewrite the whole "
                "letter in that language, keeping every fact, level, and status unchanged."
            )
        else:
            issues.append(
                "The resume is not written in the language of the original CV. "
                "Rewrite it in that language."
            )

    return ("fail" if issues else "pass"), issues


def _document_for_route(state: CareerState) -> str:
    """Pick the document produced by the current workflow."""
    if state.get("route") in COVER_LETTER_ROUTES:
        return state.get("cover_letter", "").strip()

    return state.get("tailored_resume", "").strip()


@agent_step("Truthfulness Validator")
def truthfulness_validator(state: CareerState) -> dict[str, Any]:
    document = _document_for_route(state)
    cv_text = state.get("cv_text", "").strip()
    candidate_profile = state.get("candidate_profile", {})
    candidate_evidence = state.get("candidate_evidence", [])

    if not document:
        return {"error": "Truthfulness Validator received no document."}

    if not cv_text or not candidate_evidence:
        return {"error": "Truthfulness Validator received incomplete evidence."}

    is_letter = state.get("route") in COVER_LETTER_ROUTES
    job_profile = state.get("job_profile", {})

    result = validator_chain.invoke(
        {
            "today": date.today().isoformat(),
            "name": candidate_profile.get("name") or "not stated in the CV",
            "profile": wrap_untrusted("candidate_profile", candidate_profile),
            "cv": wrap_untrusted("cv_text", cv_text),
            "evidence": wrap_untrusted("candidate_evidence", candidate_evidence),
            "job": wrap_untrusted("job_profile", job_profile),
            "gaps": wrap_untrusted("known_gaps", state.get("gap_analysis", [])),
            "request": wrap_untrusted("user_request", state.get("user_request", "")),
            "kind": "cover letter" if is_letter else "resume",
            "document": wrap_untrusted("document", document),
        }
    )

    status, issues = decide(
        result, is_letter, str(job_profile.get("posting_language") or "").strip()
    )
    update: dict[str, Any] = {
        "validation_status": status,
        "validation_issues": issues,
        "error": "",
    }

    warnings = state.get("input_warnings", [])

    if result.request_seeks_unsupported_claims and EXAGGERATION_NOTE not in warnings:
        update["input_warnings"] = [*warnings, EXAGGERATION_NOTE]

    return update
