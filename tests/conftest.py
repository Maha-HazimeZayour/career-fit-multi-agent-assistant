"""Test setup: model steps are stubbed so no Ollama is needed (Week 5, Lesson 3: evaluation)."""

import pytest
from langchain_core.runnables import RunnableLambda

import guardrails
import workflow_nodes.search_nodes as search_nodes


@pytest.fixture(autouse=True)
def stub_request_screen(monkeypatch):
    monkeypatch.setattr(
        guardrails,
        "request_screen_chain",
        RunnableLambda(lambda _: guardrails.RequestScreen(overrides_instructions=False)),
    )


@pytest.fixture(autouse=True)
def stub_search_model_steps(monkeypatch):
    monkeypatch.setattr(search_nodes, "list_tools_via_mcp", lambda: [])
    monkeypatch.setattr(
        search_nodes,
        "review_search_results",
        lambda state: {
            "verdicts": {
                number: {"number": number, "fits": True,
                         "preferred": False, "reason": ""}
                for number in range(1, len(state.get("job_search_results", [])) + 1)
            },
            "error": "",
        },
    )
