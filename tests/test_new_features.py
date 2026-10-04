"""Tests for MCP function calling, drift detection and the limitation fixes."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agents.cover_letter as cover_letter
import agents.search_query as search_query
import cli_helpers
import workflow_nodes.search_nodes as search_nodes
from monitoring import drift_report
from tools.mcp_job_client import list_tools_via_mcp

TOOL = {
    "name": "search_jobs",
    "description": "Search live remote jobs.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                      "seniority": {"type": "string"}}},
}
INTENT = {"query": "Data Analyst", "career_stage": "entry_level", "employment_type": "any"}


def test_tools_are_discovered_from_the_mcp_server():
    """Real MCP server subprocess; listing tools needs no network."""
    tools = list_tools_via_mcp()
    search = next(tool for tool in tools if tool["name"] == "search_jobs")
    properties = search["input_schema"]["properties"]
    assert {"query", "seniority", "employment_type"} <= set(properties)
    assert "Entry-level" in str(properties["seniority"])


class FakeToolModel:
    """Stands in for ChatOllama: records the tools it is given and returns `reply`."""

    def __init__(self, reply):
        self.reply, self.bound = reply, None

    def bind_tools(self, tools):
        self.bound = tools
        return RunnableLambda(lambda prompt: self.reply)


def test_agent_gets_the_mcp_tool_and_makes_the_call(monkeypatch):
    model = FakeToolModel(AIMessage(content="", tool_calls=[
        {"name": "search_jobs", "args": {"query": "junior data analyst", "seniority": "Entry-level"},
         "id": "1"}]))
    monkeypatch.setattr(search_query, "tool_model", model)

    out = search_query.call_search_tool({"job_search_intent": INTENT, "mcp_tools": [TOOL]})

    assert out["tool_call"] == {"name": "search_jobs",
                                "args": {"query": "junior data analyst", "seniority": "Entry-level"}}
    function = model.bound[0]["function"]
    assert function["name"] == "search_jobs" and "query" in function["parameters"]["properties"]


def test_a_text_answer_or_an_unknown_tool_is_not_a_call(monkeypatch):
    monkeypatch.setattr(search_query, "tool_model", FakeToolModel(AIMessage(content="Here are jobs")))
    assert search_query.call_search_tool({"job_search_intent": INTENT, "mcp_tools": [TOOL]})["tool_call"] is None

    monkeypatch.setattr(search_query, "tool_model", FakeToolModel(AIMessage(content="", tool_calls=[
        {"name": "delete_everything", "args": {}, "id": "1"}])))
    assert search_query.call_search_tool({"job_search_intent": INTENT, "mcp_tools": [TOOL]})["tool_call"] is None


def test_the_agents_arguments_are_used_within_python_limits(monkeypatch):
    monkeypatch.setattr(search_nodes, "list_tools_via_mcp", lambda: [TOOL])
    monkeypatch.setattr(search_nodes, "call_search_tool", lambda state: {
        "tool_call": {"name": "search_jobs", "args": {"query": "x" * 300, "seniority": "Boss",
                                                      "employment_type": "Intern", "count": 999}},
        "error": ""})

    arguments, caller = search_nodes.search_arguments(INTENT)

    assert caller == "agent"
    assert len(arguments["query"]) == 100
    assert arguments["seniority"] == "Entry-level"
    assert arguments["employment_type"] == "Intern"
    assert "count" not in arguments


def test_search_falls_back_to_the_intent_when_the_tool_call_fails(monkeypatch):
    monkeypatch.setattr(search_nodes, "list_tools_via_mcp", lambda: [TOOL])
    monkeypatch.setattr(search_nodes, "call_search_tool",
                        lambda state: {"error": "Job Search Query Agent (tool call) failed: offline"})
    assert search_nodes.search_arguments(INTENT) == (
        {"query": "Data Analyst", "seniority": "Entry-level", "employment_type": "", "country": ""},
        "fallback")

    def server_down():
        raise OSError("server did not start")

    monkeypatch.setattr(search_nodes, "list_tools_via_mcp", server_down)
    assert search_nodes.search_arguments(INTENT)[1] == "fallback"


def test_search_node_runs_the_agents_call_and_records_who_called(monkeypatch):
    seen = {}
    monkeypatch.setattr(search_nodes, "list_tools_via_mcp", lambda: [TOOL])
    monkeypatch.setattr(search_nodes, "call_search_tool", lambda state: {
        "tool_call": {"name": "search_jobs", "args": {"query": "junior data analyst",
                                                      "seniority": "Entry-level"}}, "error": ""})
    monkeypatch.setattr(search_nodes, "search_jobs_via_mcp",
                        lambda **kw: (seen.update(kw), [{"title": "Junior Data Analyst"}])[1])

    out = search_nodes.job_search_node({"job_search_intent": INTENT})

    assert seen == {"query": "junior data analyst", "seniority": "Entry-level",
                    "employment_type": "", "country": "", "count": 20}
    assert out["search_tool_caller"] == "agent"
    assert "Junior Data Analyst" in out["final_response"]


def test_the_country_from_the_request_reaches_the_job_api(monkeypatch):
    seen = {}
    intent = {**INTENT, "country": "Germany"}
    monkeypatch.setattr(search_nodes, "list_tools_via_mcp", lambda: [TOOL])
    monkeypatch.setattr(search_nodes, "call_search_tool", lambda state: {
        "tool_call": {"name": "search_jobs", "args": {"query": "data analyst"}}, "error": ""})
    monkeypatch.setattr(search_nodes, "search_jobs_via_mcp",
                        lambda **kw: (seen.update(kw), [{"title": "Junior Data Analyst"}])[1])

    out = search_nodes.job_search_node({"job_search_intent": intent})

    assert seen["country"] == "Germany"
    assert "Preferences applied: entry level | country: Germany" in out["final_response"]


def test_partial_matches_always_appear_as_gaps(monkeypatch):
    import agents.fit as fit

    reply = fit.FitBatch(assessments=[fit.NumberedAssessment(
        number=1, match_level="partial_match", evidence="German (B1)", evidence_type="other",
        gap_type="none", explanation="B1, not B2")])
    monkeypatch.setattr(fit, "fit_chain", RunnableLambda(lambda _: reply))

    out = fit.fit_gap_agent({"candidate_evidence": [{"evidence": "German (B1)"}],
                             "job_requirements": [{"requirement": "German B2", "priority": "preferred"}]})

    assert out["gap_analysis"] == [{"requirement": "German B2", "gap_type": "evidence",
                                    "explanation": "B1, not B2"}]


def test_validator_and_writers_get_the_date_and_the_known_gaps(monkeypatch):
    import agents.validator as validator
    from datetime import date

    seen = {}
    monkeypatch.setattr(validator, "validator_chain", RunnableLambda(
        lambda inputs: (seen.update(inputs), validator.ValidationResult(
            claims=[], identity_correct=True, document_complete=True, language_ok=True))[1]))
    validator.truthfulness_validator({"route": "cover_letter", "cover_letter": "L", "cv_text": "CV",
                                      "candidate_profile": {"name": "Rania"},
                                      "candidate_evidence": [{"e": 1}]})
    assert seen["today"] == date.today().isoformat()

    letter_inputs = {}
    fake = lambda prompt: cover_letter.CoverLetterDraft(greeting="Dear Team", paragraphs=["a", "b", "c"])
    monkeypatch.setattr(cover_letter, "cover_letter_chain", cover_letter.cover_letter_chain.steps[0]
                        | RunnableLambda(lambda p: (letter_inputs.update(text=p.to_messages()[1].content), fake(p))[1]))
    cover_letter.cover_letter_agent({"candidate_profile": {"name": "Rania"}, "job_profile": {"job_title": "Nurse"},
                                     "requirement_evidence_map": [{"requirement": "DHA licence"}],
                                     "gap_analysis": [{"requirement": "DHA licence", "gap_type": "hard_eligibility"}]})
    assert "never present these as met" in letter_inputs["text"] and "DHA licence" in letter_inputs["text"]


def test_arabic_letter_gets_no_second_comma(monkeypatch):
    fake = lambda _: cover_letter.CoverLetterDraft(
        greeting="فريق التوظيف المحترم،",
        paragraphs=["أ", "ب", "ج"], closing="مع التحية،")
    monkeypatch.setattr(cover_letter, "cover_letter_chain",
                        cover_letter.cover_letter_chain.steps[0] | RunnableLambda(fake))
    letter = cover_letter.cover_letter_agent({
        "candidate_profile": {"name": "Ada"}, "job_profile": {"posting_language": "Arabic"},
        "requirement_evidence_map": [{"requirement": "x"}]})["cover_letter"]
    assert "،," not in letter


def test_a_cv_file_given_as_the_job_posting_is_refused(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *_: "samples/cv/my_cv.docx")
    assert cli_helpers.get_job_posting() is None
    assert "must be a .txt file" in capsys.readouterr().out


def test_keyword_search_without_preferences_is_reviewed_too(monkeypatch):
    reviewed = []
    monkeypatch.setattr(search_nodes, "search_jobs_via_mcp",
                        lambda **kw: [{"title": "Marketing Manager"}, {"title": "Backend Engineer"}])
    monkeypatch.setattr(search_nodes, "review_search_results", lambda state: (
        reviewed.append(1),
        {"verdicts": {1: {"fits": True}, 2: {"fits": False, "reason": "engineering, not marketing"}},
         "error": ""})[1])

    out = search_nodes.job_search_node({"job_search_intent": {"query": "marketing"}})

    assert reviewed == [1]
    assert [job["title"] for job in out["job_search_results"]] == ["Marketing Manager"]


def run(route="resume_tailoring", seconds=200.0, retry=0, status="pass", error="",
        model="gemma4:e4b", caller=""):
    return {"route": route, "total_seconds": seconds, "steps": 7, "retry_count": retry,
            "validation_status": status, "error": error, "model": model,
            "search_tool_caller": caller}


def test_drift_needs_enough_runs():
    assert "at least 20" in drift_report([run()] * 5)


def test_stable_runs_show_no_drift():
    assert "No drift detected." in drift_report([run()] * 30)


def test_drift_flags_slower_runs_more_repairs_and_a_model_change():
    runs = [run()] * 20 + [run(seconds=320.0, retry=1, model="other-model")] * 10
    report = drift_report(runs)
    assert "Possible drift:" in report
    assert "resume_tailoring: time (s) rose from 200.0 to 320.0" in report
    assert "repair needed rose from 0% to 100%" in report
    assert "model changed" in report


def test_drift_flags_a_change_in_what_users_ask_for():
    runs = [run()] * 20 + [run(route="job_search", seconds=15.0)] * 10
    assert "requests for job_search moved from 0% to 100%" in drift_report(runs)


def test_old_log_lines_without_new_fields_still_work():
    old = {"route": "fit_analysis", "nodes": ["a", "b"], "total_seconds": 100,
           "retry_count": 0, "validation_status": "", "error": ""}
    assert "No drift detected." in drift_report([old] * 30)
