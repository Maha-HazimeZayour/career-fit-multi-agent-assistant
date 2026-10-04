"""Himalayas job API call used by the MCP server; HTML is cleaned to remove hidden instructions (Week 5, Lesson 2)."""

import html
import os
import re

import requests

JOB_API_URL = os.getenv("JOB_API_URL", "https://himalayas.app/jobs/api/search")
MAX_QUERY_CHARS = 100
SENIORITY_LEVELS = ["Entry-level", "Mid-level", "Senior", "Manager", "Director", "Executive"]
EMPLOYMENT_TYPES = ["Full Time", "Part Time", "Contractor", "Temporary", "Intern", "Volunteer", "Other"]

_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPTS = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_BLOCK_TAGS = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr)\b[^>]*>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    """Turn a job description's HTML into plain text."""
    text = _COMMENTS.sub("", text or "")
    text = _SCRIPTS.sub("", text)
    text = _BLOCK_TAGS.sub("\n", text)
    text = _TAGS.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def search_jobs_api(
    query: str = "",
    seniority: str = "",
    employment_type: str = "",
    country: str = "",
    count: int = 20,
) -> list[dict]:
    """Search live remote jobs with the free Himalayas API (no key needed)."""
    params = {"q": query.strip()[:MAX_QUERY_CHARS]}

    if seniority in SENIORITY_LEVELS:
        params["seniority"] = seniority

    if employment_type in EMPLOYMENT_TYPES:
        params["employment_type"] = employment_type

    if country.strip():
        params["country"] = country.strip()[:60]

    response = requests.get(JOB_API_URL, params=params, timeout=15)
    response.raise_for_status()

    jobs = []

    for item in response.json().get("jobs", [])[: max(1, int(count))]:
        jobs.append(
            {
                "title": item.get("title", ""),
                "company": item.get("companyName", ""),
                "location": ", ".join(item.get("locationRestrictions") or []) or "Worldwide",
                "level": ", ".join(item.get("seniority") or []),
                "job_type": item.get("employmentType", ""),
                "description": clean_html(item.get("description", "")),
                "url": item.get("applicationLink") or item.get("guid", ""),
            }
        )

    return jobs
