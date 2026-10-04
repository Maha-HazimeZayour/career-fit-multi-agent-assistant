"""Shared agent helpers: safety rules in every prompt, structured output, errors as state (Week 5, Lesson 1; Week 4, Lesson 2)."""

from functools import wraps
from typing import Any, Callable

from config import llm, llm_retry
from guardrails import SAFETY_RULES


def system_prompt(text: str) -> str:
    """Append the shared safety rules and escape braces for prompt templates."""
    full = f"{text.strip()}\n\n{SAFETY_RULES}"
    return full.replace("{", "{{").replace("}", "}}")


def structured(schema: type, model: Any = None, retry_model: Any = None) -> Any:
    """Model that returns `schema`, with a fallback for unparseable output."""
    first = model or llm
    second = retry_model or llm_retry

    return first.with_structured_output(schema).with_fallbacks(
        [second.with_structured_output(schema)]
    )


def agent_step(
    name: str,
    on_error: dict[str, Any] | None = None,
) -> Callable:
    """Turn any exception inside an agent into an `error` state update."""

    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(state: dict) -> dict:
            try:
                return function(state)
            except Exception as error:
                return {**(on_error or {}), "error": f"{name} failed: {error}"}

        return wrapper

    return decorator


def with_input_notes(state: dict, label: str, instructions: list[str]) -> dict:
    """Turn instructions the AI found hidden in a document into user-facing notes."""
    existing = list(state.get("input_warnings", []))
    found = [text.strip() for text in instructions if text and text.strip()]
    added = []

    if found:
        quoted = '"; "'.join(text[:80] for text in found[:2])
        more = f" (and {len(found) - 2} more)" if len(found) > 2 else ""
        added.append(
            f'The {label} contains text addressed to an AI ("{quoted}"{more}). '
            "It was treated as data and ignored."
        )

    new = [note for note in added if note not in existing]

    return {"input_warnings": [*existing, *new]} if new else {}
