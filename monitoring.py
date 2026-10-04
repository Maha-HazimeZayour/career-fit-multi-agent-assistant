"""Monitoring: metrics, tracing and drift detection (Week 5, Lesson 3)."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_DIR, OLLAMA_MODEL, VALIDATOR_MODEL

RECENT_RUNS = 10
SLOWER_BY = 0.25
RATE_RISE = 0.15

RATES = {
    "repair needed": lambda run: run.get("retry_count", 0) > 0,
    "validation failed": lambda run: run.get("validation_status") == "fail",
    "workflow error": lambda run: bool(run.get("error")),
    "search tool fallback": lambda run: run.get("search_tool_caller") == "fallback",
}


class RunTrace:
    def __init__(self) -> None:
        self.steps: list[dict] = []
        self._last = time.perf_counter()

    def record(self, node: str) -> None:
        now = time.perf_counter()
        self.steps.append({"node": node, "seconds": round(now - self._last, 2)})
        self._last = now

    def restart_clock(self) -> None:
        self._last = time.perf_counter()

    @property
    def nodes(self) -> list[str]:
        return [step["node"] for step in self.steps]

    @property
    def total_seconds(self) -> float:
        return round(sum(step["seconds"] for step in self.steps), 2)

    def summary(self, state: dict) -> str:
        lines = ["RUN TRACE", "Path: " + " -> ".join(self.nodes)]

        if self.steps:
            slowest = max(self.steps, key=lambda step: step["seconds"])
            lines.append(
                f"Time: {self.total_seconds}s "
                f"(slowest: {slowest['node']}, {slowest['seconds']}s)"
            )

        lines.append(
            f"Repair attempts: {state.get('retry_count', 0)} | "
            f"Validation: {state.get('validation_status', 'n/a')}"
        )

        return "\n".join(lines)

    def write_log(self, state: dict, thread_id: str) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "thread_id": thread_id,
            "route": state.get("route", ""),
            "model": OLLAMA_MODEL,
            "validator_model": VALIDATOR_MODEL,
            "nodes": self.nodes,
            "steps": len(self.nodes),
            "total_seconds": self.total_seconds,
            "retry_count": state.get("retry_count", 0),
            "validation_status": state.get("validation_status", ""),
            "warnings": len(state.get("input_warnings", [])),
            "jobs_found": len(state.get("job_search_results", [])),
            "search_tool_caller": state.get("search_tool_caller", ""),
            "error": (state.get("error") or "")[:200],
        }

        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            with (LOG_DIR / "runs.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")

        except OSError:
            pass


def load_runs(path: Path | None = None) -> list[dict]:
    path = path or LOG_DIR / "runs.jsonl"

    if not path.is_file():
        return []

    runs = []

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return runs


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def route_share(runs: list[dict], route: str) -> float:
    return average([1.0 if run.get("route") == route else 0.0 for run in runs])


def drift_report(runs: list[dict] | None = None, recent: int = RECENT_RUNS) -> str:
    runs = load_runs() if runs is None else runs

    if len(runs) < 2 * recent:
        return (
            f"DRIFT CHECK\nOnly {len(runs)} runs are logged; at least {2 * recent} "
            "are needed to compare recent runs with a baseline."
        )

    old, new = runs[:-recent], runs[-recent:]
    lines = ["DRIFT CHECK", f"Baseline: {len(old)} runs | Recent: {len(new)} runs", ""]
    flags = []

    for route in sorted({run.get("route", "") for run in new}):
        before = average([run.get("total_seconds", 0) for run in old if run.get("route") == route])
        after = average([run.get("total_seconds", 0) for run in new if run.get("route") == route])

        if before:
            lines.append(f"{route} time (s): {before:.1f} -> {after:.1f}")

            if after > before * (1 + SLOWER_BY):
                flags.append(f"{route}: time (s) rose from {before:.1f} to {after:.1f}")

    for label, test in RATES.items():
        before = average([1.0 if test(run) else 0.0 for run in old])
        after = average([1.0 if test(run) else 0.0 for run in new])
        lines.append(f"{label}: {before:.0%} -> {after:.0%}")

        if after - before > RATE_RISE:
            flags.append(f"{label} rose from {before:.0%} to {after:.0%}")

    for route in sorted({run.get("route", "") for run in runs}):
        before, after = route_share(old, route), route_share(new, route)

        if abs(after - before) > RATE_RISE:
            flags.append(f"requests for {route} moved from {before:.0%} to {after:.0%}")

    old_models = {run.get("model") for run in old if run.get("model")}
    new_models = {run.get("model") for run in new if run.get("model")}

    if old_models and new_models and old_models != new_models:
        flags.append(f"model changed: {sorted(old_models)} -> {sorted(new_models)}")

    lines.append("")
    lines.append("Possible drift:" if flags else "No drift detected.")
    lines.extend(f"- {flag}" for flag in flags)

    return "\n".join(lines)


if __name__ == "__main__":
    print(drift_report())
