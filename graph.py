"""LangGraph wiring: supervisor route, sequential handoffs, producer/critic repair loop (Week 4, Lesson 2)."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.candidate import candidate_evidence_agent
from agents.fit import fit_gap_agent
from agents.job import job_requirements_agent
from agents.search_query import job_search_query_agent
from agents.validator import truthfulness_validator
from guardrails import input_guard_node
from state import CareerState
from workflow_nodes import (
    cover_letter_node,
    failed_validation_node,
    fit_response_node,
    job_search_node,
    resume_node,
    select_job_node,
    unknown_request_node,
    validated_response_node,
    workflow_error_node,
)
from workflow_routes import (
    route_after_agent,
    route_after_candidate,
    route_after_fit,
    route_after_job_search,
    route_after_selection,
    route_after_validation,
    route_from_state,
    route_to_validator,
)


def build_graph(checkpointer=None):
    graph = StateGraph(CareerState)

    graph.add_node("input_guard", input_guard_node)

    graph.add_node("candidate", candidate_evidence_agent)
    graph.add_node("job", job_requirements_agent)
    graph.add_node("fit", fit_gap_agent)
    graph.add_node("resume", resume_node)
    graph.add_node("cover_letter", cover_letter_node)
    graph.add_node("validator", truthfulness_validator)
    graph.add_node("job_search_query", job_search_query_agent)

    graph.add_node("job_search", job_search_node)
    graph.add_node("select_job", select_job_node)
    graph.add_node("fit_response", fit_response_node)
    graph.add_node("validated", validated_response_node)
    graph.add_node("failed_validation", failed_validation_node)
    graph.add_node("workflow_error", workflow_error_node)
    graph.add_node("unknown", unknown_request_node)

    graph.add_edge(START, "input_guard")

    graph.add_conditional_edges(
        "input_guard",
        route_from_state,
        {
            "candidate": "candidate",
            "job_search": "job_search",
            "unknown": "unknown",
            "error": "workflow_error",
        },
    )

    graph.add_conditional_edges(
        "candidate",
        route_after_candidate,
        {
            "job": "job",
            "job_search_query": "job_search_query",
            "error": "workflow_error",
        },
    )

    graph.add_conditional_edges(
        "job_search_query",
        route_after_agent,
        {"continue": "job_search", "error": "workflow_error"},
    )

    graph.add_conditional_edges(
        "job_search",
        route_after_job_search,
        {"select_job": "select_job", "end": END},
    )

    graph.add_conditional_edges(
        "select_job",
        route_after_selection,
        {"job": "job", "end": END},
    )

    graph.add_conditional_edges(
        "job",
        route_after_agent,
        {"continue": "fit", "error": "workflow_error"},
    )

    graph.add_conditional_edges(
        "fit",
        route_after_fit,
        {
            "fit_response": "fit_response",
            "resume": "resume",
            "cover_letter": "cover_letter",
            "error": "workflow_error",
        },
    )

    for node in ("resume", "cover_letter"):
        graph.add_conditional_edges(
            node,
            route_to_validator,
            {"validator": "validator", "error": "workflow_error"},
        )

    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "validated": "validated",
            "resume": "resume",
            "cover_letter": "cover_letter",
            "failed_validation": "failed_validation",
            "error": "workflow_error",
        },
    )

    for node in (
        "fit_response",
        "validated",
        "failed_validation",
        "workflow_error",
        "unknown",
    ):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


career_graph = build_graph()
