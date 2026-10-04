"""Command-line entry: input guard, Supervisor routing, then the LangGraph workflow (Week 4, Lesson 2: routing to a fixed pipeline)."""

from agents.supervisor import supervisor_agent
from cli_helpers import (
    get_cv,
    get_job_posting,
    resume_run,
    start_run,
)
from config import MAX_RETRIES
from guardrails import screen_request_with_ai, screen_user_request
from workflow_nodes import unknown_request_node


def ask_job_choice(job_count: int) -> str | None:
    """Ask which job to write the letter for."""
    choice = input(
        "\nSelect a job number for the cover letter "
        "or press Enter to stop:\n> "
    ).strip()

    if not choice:
        return None

    if not choice.isdigit() or not 1 <= int(choice) <= job_count:
        print("Invalid job selection.")
        return None

    return choice


def main():
    print(
        "\n=== AI Career Fit, Resume Tailoring, "
        "and Job Search Assistant ===\n"
    )

    user_request = input("What would you like the system to do?\n> ").strip()

    problem = screen_user_request(user_request)

    if problem:
        print(problem)
        return

    problem = screen_request_with_ai(user_request)

    if problem:
        print(problem)
        return

    route_result = supervisor_agent({"user_request": user_request})
    route = route_result.get("route", "unknown")

    error = route_result.get("error", "")

    if route == "unknown" and (not error or "invalid structured output" in error):
        print(unknown_request_node({})["final_response"])
        return

    if error:
        print(error)
        return

    state = {
        "user_request": user_request,
        "request_screened": True,
        "route": route,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "error": "",
    }

    if route in {"personalized_job_search", "job_search_cover_letter"}:
        cv_text = get_cv()

        if cv_text is None:
            return

        state["cv_text"] = cv_text

    elif route in {"fit_analysis", "resume_tailoring", "cover_letter", "full_flow"}:
        cv_text = get_cv()

        if cv_text is None:
            return

        job_text = get_job_posting()

        if job_text is None:
            return

        state["cv_text"] = cv_text
        state["job_text"] = job_text

    print("\nRunning multi-agent workflow...\n")

    try:
        run = start_run(state)

        if run.interrupt:
            print("\n=== JOB SEARCH RESULTS ===\n")
            print(run.interrupt.get("message", ""))

            choice = ask_job_choice(run.interrupt.get("job_count", 0))

            if choice is None:
                return

            print("\nGenerating cover letter...\n")
            run = resume_run(run, choice)
            title = "COVER LETTER"
        else:
            title = "FINAL RESPONSE"

    except Exception as error:
        print(f"Workflow failed: {error}")
        return

    print(f"\n=== {title} ===\n")
    print(run.state.get("final_response", "No final response was produced."))

    print()
    print(run.trace.summary(run.state))


if __name__ == "__main__":
    main()
