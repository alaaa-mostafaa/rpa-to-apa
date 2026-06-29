"""
ci_stream_simulator.py
─────────────────────────────────────────────────────────────────────────────
Simulates a live GitHub Actions CI run by replaying a pre-recorded failed run
from targeted_cases.json as a sequence of fake streaming events.

Since the dataset is offline (no real-time GitHub webhook), we simulate streaming
by revealing log lines / metadata gradually — mimicking what a live CI observer
would see as jobs start and steps execute.

Usage
─────
  python ci_stream_simulator.py --run-id "apache/incubator-opendal_..."
  python ci_stream_simulator.py --list          # list all available run IDs
  python ci_stream_simulator.py --index 3       # pick case by position

Event types emitted
────────────────────
  job_started     — a CI job has begun
  step_started    — a step within a job has started
  log_chunk       — a batch of log lines arriving
  error_seen      — an ##[error] marker was spotted in the log
  step_failed     — a step has been marked failed
  run_failed      — the entire run concluded as failure
  run_success     — the entire run concluded as success (rare in this dataset)

Each event is a dict:
  {
    "type": "log_chunk",
    "run_id": "...",
    "job_file": "build.txt",
    "step_label": "Run tests",
    "chunk_index": 4,
    "text": "pytest failed..."
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional

# -- Windows UTF-8 output fix ------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# ----------------------------------------------------------------------------

# ─── configuration ────────────────────────────────────────────────────────────

DATA_FILE = Path(__file__).with_name("targeted_cases.json")

# How many log lines to emit per log_chunk event
LINES_PER_CHUNK = 3

# Simulated inter-event delay (seconds).  0 = instant replay.
DEFAULT_DELAY = 0.05

# Maximum log chunks to emit per step (avoids flooding on very long logs)
MAX_CHUNKS_PER_STEP = 30


# ─── loading ──────────────────────────────────────────────────────────────────

def load_cases(path: Path = DATA_FILE) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_run(cases: List[dict], run_id: str) -> Optional[dict]:
    for c in cases:
        if c.get("intake", {}).get("run_id") == run_id:
            return c
    return None


def list_runs(cases: List[dict]) -> None:
    print(f"{'#':<4}  {'Conclusion':<10}  {'Run ID'}")
    print("-" * 90)
    for i, c in enumerate(cases):
        intake = c.get("intake", {})
        rid = intake.get("run_id", "?")
        conclusion = intake.get("conclusion", "?")
        print(f"{i:<4}  {conclusion:<10}  {rid}")


# ─── synthetic log generation ─────────────────────────────────────────────────
#
# The targeted_cases.json stores only sample_error_lines (the extracted
# highlights), not the full raw log text.  We reconstruct a plausible
# synthetic log from available metadata so the simulator can produce
# realistic streaming chunks.

def _build_synthetic_log_lines(case: dict) -> List[str]:
    """
    Build a synthetic ordered list of log lines for a case.

    Structure:
      1. Job header (##[group]Set up job)
      2. Per-step preamble lines
      3. Injected actual sample_error_lines near the end
      4. ##[error] markers
      5. Job footer (Process completed with exit code N)
    """
    intake = case.get("intake", {})
    extraction = case.get("extraction", {})
    classification = case.get("classification", {})
    ground_truth = case.get("ground_truth", {})

    repo = intake.get("repo", "unknown/repo")
    branch = intake.get("branch", "main")
    workflow = intake.get("workflow", "ci.yml")
    commit_sha = intake.get("commit_sha", "abc1234")
    commit_title = intake.get("commit_title", "some commit")
    event = intake.get("event", "push")
    runner = "ubuntu-latest"

    sample_errors: List[str] = extraction.get("sample_error_lines", [])
    strategies: List[str] = extraction.get("strategies", [])
    n_steps = extraction.get("total_steps_extracted", 1)

    # Evidence lines from classification (helps simulate "middle" of the log)
    evidence_lines: List[str] = classification.get("evidence", []) or []

    lines: List[str] = []

    # ── 1. Job preamble ──────────────────────────────────────────────────────
    lines.append(f"##[group]Set up job")
    lines.append(f"Current runner version: '2.308.0'")
    lines.append(f"Operating System: Ubuntu 22.04.3 LTS")
    lines.append(f"Runner Image: {runner}")
    lines.append(f"##[endgroup]")
    lines.append(f"##[group]Checkout {repo}")
    lines.append(f"Syncing repository: {repo}")
    lines.append(f"Getting Git version info")
    lines.append(f"Initializing the repository")
    lines.append(f"Setting up auth")
    lines.append(f"##[endgroup]")

    # ── 2. Workflow / event context lines ────────────────────────────────────
    lines.append(f"##[group]Run workflow: {workflow}")
    lines.append(f"Event: {event}")
    lines.append(f"Branch: {branch}")
    lines.append(f"Commit: {commit_sha} — {commit_title}")
    lines.append(f"##[endgroup]")

    # ── 3. Simulated step outputs ────────────────────────────────────────────
    step_labels = _infer_step_labels(case, n_steps)

    for step_idx, step_label in enumerate(step_labels):
        lines.append(f"##[group]Run {step_label}")
        lines.append(f"  shell: /usr/bin/bash -e {{0}}")
        lines.append(f"  workdir: /home/runner/work/{repo.split('/')[-1]}")

        is_last_step = step_idx == len(step_labels) - 1

        if is_last_step:
            # Inject evidence lines that hint at the actual error
            for ev_line in evidence_lines:
                # Strip "log: " prefix if present
                if ev_line.lower().startswith("log:"):
                    ev_line = ev_line[4:].strip()
                lines.append(f"  {ev_line}")

            # Inject the actual sample error lines BEFORE the ##[error] markers
            for err_line in sample_errors:
                if not err_line.startswith("##[error]"):
                    lines.append(f"  {err_line}")

            # Inject ##[error] markers
            for err_line in sample_errors:
                if err_line.startswith("##[error]"):
                    lines.append(err_line)

        else:
            # Non-failing steps
            lines.append(f"  [OK] {step_label} completed")
            lines.append(f"  Duration: {2 + step_idx}s")

        lines.append(f"##[endgroup]")

    # ── 4. Run conclusion ────────────────────────────────────────────────────
    conclusion = intake.get("conclusion", "failure")
    exit_code = 1 if conclusion == "failure" else 0
    lines.append(f"##[group]Complete job")
    lines.append(f"Finishing: Complete job")
    lines.append(f"##[endgroup]")

    return lines


def _infer_step_labels(case: dict, n_steps: int) -> List[str]:
    """
    Guess step labels from available metadata.
    Uses classification evidence or generic labels as fallback.
    """
    category = case.get("classification", {}).get("category", "")
    workflow = case.get("intake", {}).get("workflow", "ci.yml")

    # Try to produce realistic labels based on category + workflow
    label_map = {
        "CODE_REGRESSION": ["Checkout", "Build", "Run tests"],
        "DEPENDENCY_CONFLICT": ["Checkout", "Install dependencies", "Build"],
        "CONFIG_ERROR": ["Checkout", "Validate configuration", "Run workflow"],
        "ENV_FLAKINESS": ["Checkout", "Set up environment", "Run job"],
        "TEST_FLAKINESS": ["Checkout", "Build", "Run tests", "Report"],
        "INFRA_INCOMPATIBILITY": ["Checkout", "Set up runner", "Build", "Deploy"],
        "TOOLING_ARTIFACT": ["Checkout", "Build"],
    }

    labels = label_map.get(category, ["Checkout", "Build", "Run"])

    # Trim or extend to match n_steps
    if n_steps > len(labels):
        labels += [f"Step {i+1}" for i in range(len(labels), n_steps)]
    elif n_steps > 0:
        labels = labels[:n_steps]

    if not labels:
        labels = ["Run"]

    return labels


# ─── event generator ──────────────────────────────────────────────────────────

def stream_run_events(
    case: dict,
    delay: float = DEFAULT_DELAY,
) -> Generator[dict, None, None]:
    """
    Yield fake live CI events for a case, one at a time.

    Each yielded value is an event dict with at minimum:
        type, run_id, timestamp_sim (simulated elapsed seconds)
    """
    intake = case.get("intake", {})
    run_id = intake.get("run_id", "unknown")
    conclusion = intake.get("conclusion", "failure")
    n_jobs = intake.get("n_jobs", 1)
    failed_jobs_count = intake.get("failed_jobs_count", 1)

    sim_time = 0.0  # simulated elapsed seconds

    def tick(dt: float = 1.5) -> float:
        nonlocal sim_time
        sim_time += dt
        return sim_time

    # ── Event 0: run_started ─────────────────────────────────────────────────
    yield {
        "type": "run_started",
        "run_id": run_id,
        "workflow": intake.get("workflow", "ci.yml"),
        "repo": intake.get("repo", "?"),
        "branch": intake.get("branch", "?"),
        "event_trigger": intake.get("event", "push"),
        "commit_sha": intake.get("commit_sha", "?"),
        "commit_title": intake.get("commit_title", "?"),
        "timestamp_sim": tick(0),
    }
    if delay:
        time.sleep(delay)

    # ── Per-job events ───────────────────────────────────────────────────────
    log_lines = _build_synthetic_log_lines(case)
    step_labels = _infer_step_labels(case, case.get("extraction", {}).get("total_steps_extracted", 1))
    sample_errors = case.get("extraction", {}).get("sample_error_lines", [])
    error_markers = [ln for ln in sample_errors if ln.startswith("##[error]")]

    for job_idx in range(max(n_jobs, 1)):
        job_file = f"job_{job_idx + 1}.txt"  # synthetic filename

        # job_started
        yield {
            "type": "job_started",
            "run_id": run_id,
            "job_index": job_idx,
            "job_file": job_file,
            "timestamp_sim": tick(0.5),
        }
        if delay:
            time.sleep(delay)

        # Emit per-step events
        for step_idx, step_label in enumerate(step_labels):
            is_last = step_idx == len(step_labels) - 1

            # step_started
            yield {
                "type": "step_started",
                "run_id": run_id,
                "job_file": job_file,
                "step_index": step_idx,
                "step_label": step_label,
                "timestamp_sim": tick(0.8),
            }
            if delay:
                time.sleep(delay)

            # log_chunk events — split log lines into chunks
            # Gather the lines that belong to this step from our synthetic log
            step_lines = _extract_step_lines(log_lines, step_label)

            chunk_idx = 0
            for i in range(0, min(len(step_lines), LINES_PER_CHUNK * MAX_CHUNKS_PER_STEP), LINES_PER_CHUNK):
                chunk = step_lines[i : i + LINES_PER_CHUNK]
                text = "\n".join(chunk)

                yield {
                    "type": "log_chunk",
                    "run_id": run_id,
                    "job_file": job_file,
                    "step_label": step_label,
                    "chunk_index": chunk_idx,
                    "text": text,
                    "timestamp_sim": tick(0.3),
                }
                chunk_idx += 1
                if delay:
                    time.sleep(delay)

                # Check if this chunk contains an ##[error] marker
                for marker in error_markers:
                    if marker in text or any(marker in ln for ln in chunk):
                        yield {
                            "type": "error_seen",
                            "run_id": run_id,
                            "job_file": job_file,
                            "step_label": step_label,
                            "error_text": marker,
                            "timestamp_sim": tick(0.1),
                        }
                        if delay:
                            time.sleep(delay)
                        break

            # step_failed (only for the last step if run is a failure)
            is_failed_job = job_idx < failed_jobs_count
            if is_last and is_failed_job and conclusion == "failure":
                yield {
                    "type": "step_failed",
                    "run_id": run_id,
                    "job_file": job_file,
                    "step_label": step_label,
                    "exit_code": 1,
                    "timestamp_sim": tick(0.2),
                }
                if delay:
                    time.sleep(delay)

        # job_failed / job_succeeded
        is_failed_job = job_idx < failed_jobs_count and conclusion == "failure"
        yield {
            "type": "job_failed" if is_failed_job else "job_succeeded",
            "run_id": run_id,
            "job_file": job_file,
            "timestamp_sim": tick(0.5),
        }
        if delay:
            time.sleep(delay)

    # ── Final run conclusion ─────────────────────────────────────────────────
    yield {
        "type": "run_failed" if conclusion == "failure" else "run_succeeded",
        "run_id": run_id,
        "conclusion": conclusion,
        "failed_jobs": failed_jobs_count,
        "timestamp_sim": tick(1.0),
    }


def _extract_step_lines(log_lines: List[str], step_label: str) -> List[str]:
    """
    Extract the lines belonging to a specific step group from the synthetic log.
    Looks for ##[group]Run <step_label> ... ##[endgroup].
    """
    inside = False
    result: List[str] = []
    for line in log_lines:
        if f"##[group]Run {step_label}" in line:
            inside = True
            result.append(line)
            continue
        if inside:
            if line.strip() == "##[endgroup]":
                result.append(line)
                break
            result.append(line)
    # Fallback: just return all lines if we couldn't find the group
    if not result:
        result = log_lines
    return result


# ─── CLI pretty-printer ───────────────────────────────────────────────────────

EVENT_ICONS = {
    "run_started":    ">>",
    "job_started":    "JOB",
    "step_started":   "->",
    "log_chunk":      "  |",
    "error_seen":     "ERR",
    "step_failed":    "FAIL",
    "job_failed":     "[X]",
    "job_succeeded":  "[OK]",
    "run_failed":     "DEAD",
    "run_succeeded":  "DONE",
}


def print_event(idx: int, event: dict) -> None:
    etype = event["type"]
    icon = EVENT_ICONS.get(etype, "·")
    sim_t = event.get("timestamp_sim", 0)
    prefix = f"[{idx:>3}] t={sim_t:>5.1f}s  {icon}  {etype:<18}"

    if etype == "run_started":
        extra = f"{event.get('repo')} | branch={event.get('branch')} | {event.get('commit_title', '')[:50]}"
    elif etype == "job_started":
        extra = event.get("job_file", "")
    elif etype == "step_started":
        extra = f"{event.get('job_file')} -> step[{event.get('step_index')}] '{event.get('step_label')}'"
    elif etype == "log_chunk":
        text = event.get("text", "")
        # Shorten the preview
        preview = text.replace("\n", " | ")[:80]
        extra = f"chunk#{event.get('chunk_index')}  {preview}"
    elif etype == "error_seen":
        extra = f"{event.get('error_text', '')[:100]}"
    elif etype == "step_failed":
        extra = f"'{event.get('step_label')}' exit_code={event.get('exit_code')}"
    elif etype in ("job_failed", "job_succeeded"):
        extra = event.get("job_file", "")
    elif etype in ("run_failed", "run_succeeded"):
        extra = f"failed_jobs={event.get('failed_jobs', 0)}"
    else:
        extra = ""

    print(f"{prefix}  {extra}")


# -- Windows safe print -------------------------------------------------------
def _safe_print(s: str) -> None:
    """Print replacing any unrepresentable chars."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a CI run as fake live streaming events."
    )
    parser.add_argument(
        "--run-id",
        metavar="RUN_ID",
        help="Exact run_id to replay (from targeted_cases.json).",
    )
    parser.add_argument(
        "--index",
        type=int,
        metavar="N",
        help="Pick case by 0-based index (alternative to --run-id).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available run IDs and exit.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        metavar="SEC",
        help=f"Seconds to pause between events (default {DEFAULT_DELAY}). 0 = instant.",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        default=False,
        help="When using --list or default selection, only show failed runs.",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_FILE),
        metavar="PATH",
        help=f"Path to targeted_cases.json (default: {DATA_FILE}).",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.data))

    if args.failed_only:
        cases = [c for c in cases if c.get("intake", {}).get("conclusion") == "failure"]

    if args.list:
        list_runs(cases)
        return

    # Select case
    if args.run_id:
        case = find_run(cases, args.run_id)
        if case is None:
            print(f"ERROR: run_id not found: {args.run_id}", file=sys.stderr)
            print("Use --list to see available run IDs.", file=sys.stderr)
            sys.exit(1)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(cases):
            print(f"ERROR: index {args.index} out of range (0–{len(cases)-1})", file=sys.stderr)
            sys.exit(1)
        case = cases[args.index]
    else:
        # Default: pick the first failed run
        failed = [c for c in cases if c.get("intake", {}).get("conclusion") == "failure"]
        if not failed:
            print("ERROR: No failed runs found in dataset.", file=sys.stderr)
            sys.exit(1)
        case = failed[0]
        print(f"(No --run-id specified, using first failed run)\n")

    # Print header
    intake = case.get("intake", {})
    print("=" * 70)
    print(f"  STREAM REPLAY: {intake.get('run_id', '?')}")
    print(f"  repo:       {intake.get('repo', '?')}")
    print(f"  workflow:   {intake.get('workflow', '?')}")
    print(f"  branch:     {intake.get('branch', '?')}")
    print(f"  conclusion: {intake.get('conclusion', '?')}")
    print(f"  delay:      {args.delay}s between events")
    print("=" * 70)
    print()

    event_count = 0
    for event in stream_run_events(case, delay=args.delay):
        print_event(event_count + 1, event)
        event_count += 1

    print()
    print(f"─── Replay complete: {event_count} events emitted ───")


if __name__ == "__main__":
    main()
