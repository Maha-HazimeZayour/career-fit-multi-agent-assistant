"""Fit and Gap agent: compares each requirement with the CV evidence (Week 4, Lesson 2)."""

from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agents.common import agent_step, structured, system_prompt
from config import FIT_GROUP_SIZE, FIT_MAX_CONCURRENCY
from guardrails import wrap_untrusted
from state import CareerState


FIT_PROMPT = """
You are the Career Fit and Gap Agent.

Compare each numbered job requirement with verified candidate evidence.
Assess every requirement separately and return one assessment per number.

Use only supplied evidence. Do not invent, strengthen, or infer missing
experience from a related role, industry, education, or context.

MATCH
- strong_match: evidence clearly supports the requirement
- partial_match: relevant evidence exists but does not fully support it
- gap: evidence is insufficient

Academic/project evidence may support skills or knowledge, but must not be
treated as professional experience.

If a specific responsibility or experience is not directly supported, do not
assume the candidate performed it.

GAP TYPE
- none
- presentation
- evidence
- learnable_skill
- professional_experience
- hard_eligibility

Use hard_eligibility only for required non-substitutable conditions such as
licenses, certifications, language levels, work authorization, or eligibility.
Preferred requirements must never be hard_eligibility.

If no supporting evidence exists, return an empty evidence string and
evidence_type "none".

JOB CONDITIONS THE CV CANNOT SHOW
Requirements about the job itself (full-time or part-time, on-site or remote,
work location, start date, salary, working hours) are not skills. A CV usually
does not state them. Return gap, evidence_type "none", gap_type "evidence" and
the explanation "not stated in the CV". Never call them partial_match, and never
guess from unrelated details.

Be conservative when evidence is unclear.

KEEP ANSWERS SHORT
evidence: a short quote or phrase (under 15 words).
explanation: one short sentence.
"""


class RequirementAssessment(BaseModel):
    match_level: Literal["strong_match", "partial_match", "gap"]
    evidence: str
    evidence_type: Literal[
        "professional",
        "academic",
        "project",
        "certification",
        "other",
        "none",
    ]
    gap_type: Literal[
        "none",
        "presentation",
        "evidence",
        "learnable_skill",
        "professional_experience",
        "hard_eligibility",
    ]
    explanation: str


class NumberedAssessment(RequirementAssessment):
    number: int


class FitBatch(BaseModel):
    assessments: list[NumberedAssessment]


fit_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt(FIT_PROMPT)),
            (
                "human",
                "JOB REQUIREMENTS:\n{requirements}\n\n"
                "CANDIDATE EVIDENCE:\n{evidence}",
            ),
        ]
    )
    | structured(FitBatch)
)


@agent_step("Career Fit and Gap Agent")
def fit_gap_agent(state: CareerState) -> dict[str, Any]:
    candidate_evidence = state.get("candidate_evidence", [])
    job_requirements = state.get("job_requirements", [])

    if not candidate_evidence:
        return {"error": "Career Fit and Gap Agent received no candidate evidence."}

    if not job_requirements:
        return {"error": "Career Fit and Gap Agent received no job requirements."}

    evidence_block = wrap_untrusted("candidate_evidence", candidate_evidence)

    by_number: dict[int, RequirementAssessment] = {}

    for _attempt in range(2):
        missing = [
            n for n in range(1, len(job_requirements) + 1) if n not in by_number
        ]

        if not missing:
            break

        groups = [
            missing[i : i + FIT_GROUP_SIZE]
            for i in range(0, len(missing), FIT_GROUP_SIZE)
        ]
        results = fit_chain.batch(
            [
                {
                    "requirements": "\n".join(
                        f"{n}. {job_requirements[n - 1].get('requirement', '')}"
                        f" (priority: {job_requirements[n - 1].get('priority', 'unclear')})"
                        for n in group
                    ),
                    "evidence": evidence_block,
                }
                for group in groups
            ],
            config={"max_concurrency": FIT_MAX_CONCURRENCY},
        )

        for group, batch in zip(groups, results):
            for item in batch.assessments:
                if item.number in group:
                    by_number.setdefault(item.number, item)

    assessments = [
        by_number.get(n)
        or RequirementAssessment(
            match_level="gap",
            evidence="",
            evidence_type="none",
            gap_type="evidence",
            explanation="This requirement could not be assessed automatically.",
        )
        for n in range(1, len(job_requirements) + 1)
    ]

    requirement_evidence_map = []
    gap_analysis = []

    for requirement, result in zip(job_requirements, assessments):
        assessment = result.model_dump()
        priority = requirement.get("priority", "unclear")

        if priority != "required" and assessment["gap_type"] == "hard_eligibility":
            assessment["gap_type"] = "evidence"

        if assessment["match_level"] != "strong_match" and assessment["gap_type"] == "none":
            assessment["gap_type"] = "evidence"

        match = {
            "requirement": requirement.get("requirement", ""),
            "priority": priority,
            **assessment,
        }

        requirement_evidence_map.append(match)

        if match["gap_type"] != "none":
            gap_analysis.append(
                {
                    "requirement": match["requirement"],
                    "gap_type": match["gap_type"],
                    "explanation": match["explanation"],
                }
            )

    return {
        "requirement_evidence_map": requirement_evidence_map,
        "gap_analysis": gap_analysis,
        "error": "",
    }
