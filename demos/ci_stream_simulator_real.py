"""
ci_stream_simulator_real.py
─────────────────────────────────────────────────────────────────────────────
Streams REAL full_log.txt files from comparison_results/balanced_50_full_logs
as live CI events, feeding them line-by-line into the monitor.

This replaces the synthetic simulator entirely for the 50 cases that have
real logs. No fake lines are invented. Every chunk the monitor sees came
directly from the original GitHub Actions log tarball.

Key insight from dataset analysis:
  - 23/25 failure cases have ##[error] markers
  - First error appears at 9.8% – 97.5% through the log (avg 40.9%)
  - Success cases have 0 ##[error] in 23/25 cases
  - Both success and failure logs contain real warnings (noise)

Usage
─────
  python ci_stream_simulator_real.py --list
  python ci_stream_simulator_real.py --index 0          # first failure case
  python ci_stream_simulator_real.py --index 25         # first success case
  python ci_stream_simulator_real.py --case 001         # by case number
  python ci_stream_simulator_real.py --delay 0          # instant replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── configuration ────────────────────────────────────────────────────────────

REAL_LOGS_DIR = (
    Path(__file__).resolve().parents[1]
    / "comparison_results"
    / "balanced_50_full_logs_20260516"
)

# Lines per chunk event — same as synthetic simulator
LINES_PER_CHUNK = 10

# Default delay between events (seconds)
DEFAULT_DELAY = 0.0

JOB_SEPARATOR = "=" * 100


# ─── loading ──────────────────────────────────────────────────────────────────

def load_manifest(base: Path = REAL_LOGS_DIR) -> List[dict]:
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    # Fall back: scan directories
    cases = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta["_dir"] = str(d)
            cases.append(meta)
    return cases


def load_real_case_dirs(base: Path = REAL_LOGS_DIR) -> List[dict]:
    """Return list of case dicts sorted by directory name."""
    cases = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        log_path = d / "full_log.txt"
        meta_path = d / "metadata.json"
        if not log_path.exists():
            continue
        meta = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        meta["_dir"] = str(d)
        meta["_log_path"] = str(log_path)
        meta["_case_name"] = d.name
        meta["_label"] = "failure" if "failure" in d.name else "success"
        cases.append(meta)
    return cases


def find_case(cases: List[dict], identifier: str) -> Optional[dict]:
    """Find case by case number prefix (e.g. '001') or run_id."""
    for c in cases:
        if c["_case_name"].startswith(identifier):
            return c
        if c.get("run_id", "") == identifier:
            return c
    return None


def list_cases(cases: List[dict]) -> None:
    print(f"{'#':<4}  {'label':<8}  {'lines':>7}  {'run_id'}")
    print("-" * 90)
    for c in cases:
        log_path = Path(c["_log_path"])
        try:
            n_lines = sum(1 for _ in open(log_path, encoding="utf-8", errors="replace"))
        except Exception:
            n_lines = 0
        print(f"{c['_case_name'][:3]:<4}  {c['_label']:<8}  {n_lines:>7,}  {c.get('run_id', '?')}")


# ─── line parsing ─────────────────────────────────────────────────────────────

def parse_log_line(raw: str) -> str:
    """
    GitHub Actions log lines look like:
      2023-11-09T02:12:04.7495732Z ##[error]Process completed...
    Strip the timestamp prefix if present, return the rest.
    """
    # Timestamps are ISO-format: digits-T digits with Z at end
    if len(raw) > 29 and raw[4] == "-" and raw[10] == "T" and raw[28:30] in ("Z ", "Z\t"):
        return raw[29:].rstrip()
    return raw.rstrip()


def detect_job_boundary(line: str) -> Optional[str]:
    """Detect the JOB LOG separator inserted by the comparison script."""
    stripped = line.strip()
    if stripped.startswith("JOB LOG") and ":" in stripped:
        # e.g. "JOB LOG 1: Build and save artifacts/12_Post Run.txt"
        return stripped.split(":", 1)[1].strip()
    return None


# ─── event streaming ──────────────────────────────────────────────────────────

def stream_real_events(
    case: dict,
    delay: float = DEFAULT_DELAY,
) -> Generator[dict, None, None]:
    """
    Read the real full_log.txt line-by-line and yield CI events:
      run_started → [job_started → log_chunk...] × N → run_failed/succeeded
    """
    run_id = case.get("run_id", "?")
    label = case["_label"]
    log_path = case["_log_path"]

    sim_time = 0.0

    def tick(dt: float = 0.3) -> float:
        nonlocal sim_time
        sim_time += dt
        return sim_time

    # ── run_started ──────────────────────────────────────────────────────────
    yield {
        "type": "run_started",
        "run_id": run_id,
        "workflow": case.get("workflow_path", "?").split("/")[-1],
        "repo": case.get("repo", "?"),
        "branch": "?",                   # not in metadata; kept generic
        "event_trigger": "push",
        "commit_sha": "?",
        "commit_title": "?",
        "timestamp_sim": tick(0),
        "_source": "real_log",
    }
    if delay:
        time.sleep(delay)

    # ── stream log file ──────────────────────────────────────────────────────
    current_job: str = "job_1.txt"
    job_index: int = 0
    current_job_lines: List[str] = []
    chunk_idx: int = 0
    seen_jobs: set = set()
    error_markers_seen: set = set()

    def flush_chunk() -> Optional[dict]:
        """Package accumulated lines as a log_chunk event."""
        nonlocal chunk_idx
        if not current_job_lines:
            return None
        text = "\n".join(current_job_lines)
        current_job_lines.clear()
        ev = {
            "type": "log_chunk",
            "run_id": run_id,
            "job_file": current_job,
            "step_label": "?",
            "chunk_index": chunk_idx,
            "text": text,
            "timestamp_sim": tick(0.3),
        }
        chunk_idx += 1
        return ev

    def emit_job_started(job_name: str) -> dict:
        return {
            "type": "job_started",
            "run_id": run_id,
            "job_index": job_index,
            "job_file": job_name,
            "timestamp_sim": tick(0.5),
        }

    with open(log_path, encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    for raw in raw_lines:
        line = parse_log_line(raw)
        if not line:
            continue

        # Detect job boundary separator
        if "=" * 60 in line:
            continue
        job_name = detect_job_boundary(line)
        if job_name:
            # Flush any pending chunk from previous job
            ev = flush_chunk()
            if ev:
                yield ev
                if delay:
                    time.sleep(delay)

            current_job = job_name.replace("/", "_").replace(" ", "_")[:40] + ".txt"
            job_index += 1
            chunk_idx = 0
            if current_job not in seen_jobs:
                seen_jobs.add(current_job)
                yield emit_job_started(current_job)
                if delay:
                    time.sleep(delay)
            continue

        # Accumulate lines
        current_job_lines.append(line)

        # Check for error markers in this line BEFORE it gets chunked
        if "##[error]" in line:
            error_key = line.strip()[:80]
            if error_key not in error_markers_seen:
                error_markers_seen.add(error_key)
                # First: flush what we have up to and including this error line
                ev = flush_chunk()
                if ev:
                    yield ev
                    if delay:
                        time.sleep(delay)
                # Then emit the error_seen event
                yield {
                    "type": "error_seen",
                    "run_id": run_id,
                    "job_file": current_job,
                    "step_label": "?",
                    "error_text": line.strip(),
                    "timestamp_sim": tick(0.1),
                }
                if delay:
                    time.sleep(delay)

        # Emit chunk when buffer is full
        elif len(current_job_lines) >= LINES_PER_CHUNK:
            ev = flush_chunk()
            if ev:
                yield ev
                if delay:
                    time.sleep(delay)

    # Flush final lines
    ev = flush_chunk()
    if ev:
        yield ev
        if delay:
            time.sleep(delay)

    # ── run conclusion ───────────────────────────────────────────────────────
    conclusion_type = "run_failed" if label == "failure" else "run_succeeded"
    yield {
        "type": conclusion_type,
        "run_id": run_id,
        "conclusion": label,
        "failed_jobs": len(error_markers_seen),
        "timestamp_sim": tick(1.0),
        "_source": "real_log",
    }


# ─── CLI pretty-printer ───────────────────────────────────────────────────────

EVENT_ICONS = {
    "run_started":   ">>",
    "job_started":   "JOB",
    "step_started":  "->",
    "log_chunk":     "  |",
    "error_seen":    "ERR",
    "step_failed":   "FAIL",
    "job_failed":    "[X]",
    "job_succeeded": "[OK]",
    "run_failed":    "DEAD",
    "run_succeeded": "DONE",
}


def print_event(idx: int, event: dict) -> None:
    etype = event["type"]
    icon = EVENT_ICONS.get(etype, "  ·")
    sim_t = event.get("timestamp_sim", 0)
    prefix = f"[{idx:>4}] t={sim_t:>7.1f}s  {icon}  {etype:<18}"

    if etype == "run_started":
        extra = f"{event.get('repo')}  |  {event.get('workflow')}"
    elif etype == "job_started":
        extra = event.get("job_file", "")
    elif etype == "log_chunk":
        text = event.get("text", "")
        preview = text.replace("\n", " | ")[:90]
        extra = f"chunk#{event.get('chunk_index')}  {preview}"
    elif etype == "error_seen":
        extra = event.get("error_text", "")[:100]
    elif etype in ("run_failed", "run_succeeded"):
        extra = f"error_markers_seen={event.get('failed_jobs', 0)}"
    else:
        extra = ""

    print(f"{prefix}  {extra}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream REAL CI log files as live events."
    )
    parser.add_argument("--list", action="store_true", help="List available cases.")
    parser.add_argument("--index", type=int, metavar="N",
                        help="Pick case by 0-based index (0-24 = failure, 25-49 = success).")
    parser.add_argument("--case", metavar="NNN",
                        help="Pick by case number prefix (e.g. '001').")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help="Seconds between events (default 0 = instant).")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--successes-only", action="store_true")
    args = parser.parse_args()

    cases = load_real_case_dirs(REAL_LOGS_DIR)
    if args.failures_only:
        cases = [c for c in cases if c["_label"] == "failure"]
    elif args.successes_only:
        cases = [c for c in cases if c["_label"] == "success"]

    if args.list:
        list_cases(cases)
        return

    if args.case:
        case = find_case(cases, args.case)
        if not case:
            print(f"ERROR: case '{args.case}' not found", file=sys.stderr)
            sys.exit(1)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(cases):
            print(f"ERROR: index {args.index} out of range 0-{len(cases)-1}", file=sys.stderr)
            sys.exit(1)
        case = cases[args.index]
    else:
        case = cases[0]
        print(f"(No case specified, using: {case['_case_name']})\n")

    # Header
    log_path = Path(case["_log_path"])
    n_lines = sum(1 for _ in open(log_path, encoding="utf-8", errors="replace"))
    print("=" * 70)
    print(f"  REAL LOG REPLAY: {case['_case_name']}")
    print(f"  run_id:    {case.get('run_id', '?')}")
    print(f"  label:     {case['_label'].upper()}")
    print(f"  log lines: {n_lines:,}")
    print(f"  delay:     {args.delay}s")
    print("=" * 70)
    print()

    event_count = 0
    for event in stream_real_events(case, delay=args.delay):
        print_event(event_count + 1, event)
        event_count += 1

    print()
    print(f"--- Replay complete: {event_count} events from {n_lines:,} real log lines ---")


if __name__ == "__main__":
    main()
