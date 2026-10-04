"""Reads the inputs and runs the graph with a pause for the user's job choice (Week 5, Lesson 2: human approval)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from config import RECURSION_LIMIT
from graph import career_graph
from monitoring import RunTrace
from tools.cv_reader import read_cv

END_MARKER = "END"


def read_text_file(file_path: str) -> str:
    path = Path(file_path).expanduser()

    if not path.is_file():
        print(f"Error: file not found at '{file_path}'.")
        return ""

    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as error:
        print(f"Error while reading the file: {error}")
        return ""


def get_multiline_input(prompt: str) -> str:
    """Read pasted text until a line that only says END (or Ctrl-D)."""
    print(prompt)
    print(f"(Type {END_MARKER} on its own line when finished.)")

    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip().upper() == END_MARKER:
            break

        lines.append(line)

    return "\n".join(lines).strip()


def get_cv() -> str | None:
    cv_path = input("\nEnter the path to the candidate CV:\n> ").strip()

    if not cv_path:
        print("No CV path provided.")
        return None

    cv_text = read_cv.invoke({"file_path": cv_path})

    if cv_text.startswith("Error:"):
        print(cv_text)
        return None

    return cv_text


def get_job_posting() -> str | None:
    job_path = input(
        "\nEnter the path to the job posting TXT file "
        "or press Enter to paste it manually:\n> "
    ).strip()

    if job_path:
        if Path(job_path).suffix.lower() != ".txt":
            print(
                "The job posting must be a .txt file. If you entered the CV "
                "here by mistake, start again and give the job posting file."
            )
            return None

        job_text = read_text_file(job_path)
    else:
        job_text = get_multiline_input("\nPaste the job posting below.")

    if not job_text:
        print("No job posting provided.")
        return None

    return job_text


@dataclass
class RunResult:
    state: dict[str, Any]
    thread_id: str
    trace: RunTrace
    interrupt: dict[str, Any] | None = None


def _stream(graph_input: Any, thread_id: str, trace: RunTrace) -> RunResult:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    for chunk in career_graph.stream(graph_input, config=config, stream_mode="updates"):
        for node in chunk:
            if node != "__interrupt__":
                trace.record(node)

    snapshot = career_graph.get_state(config)
    pending = None

    for task in snapshot.tasks:
        if task.interrupts:
            pending = task.interrupts[0].value
            break

    result = RunResult(
        state=dict(snapshot.values),
        thread_id=thread_id,
        trace=trace,
        interrupt=pending,
    )

    if pending is None:
        trace.write_log(result.state, thread_id)

    return result


def start_run(state: dict) -> RunResult:
    """Run the graph until it finishes or pauses for the candidate's choice."""
    return _stream(state, str(uuid4()), RunTrace())


def resume_run(run: RunResult, answer: str) -> RunResult:
    """Continue a paused run with the candidate's answer."""
    run.trace.restart_clock()
    return _stream(Command(resume=answer), run.thread_id, run.trace)
