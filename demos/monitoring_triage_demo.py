"""
monitoring_triage_demo.py
─────────────────────────────────────────────────────────────────────────────
End-to-end demonstration of the continuous monitoring prototype.

Flow
────
  1. Load a selected failed run from targeted_cases.json
  2. Start the stream simulator (ci_stream_simulator.py)
  3. For each streaming event:
       a. Feed it to the CI monitor (ci_monitor.py)
       b. Print the current risk score after each update
       c. If risk >= TRIAGE_THRESHOLD and not yet triaged:
            → Build a partial_raw_run from accumulated monitor state
            → Call run_agent(partial_raw_run) — the real APA
            → Print early triage result
  4. After run_failed event: run full triage on complete evidence
  5. Print a side-by-side comparison: early vs final triage

Usage
─────
  python monitoring_triage_demo.py
  python monitoring_triage_demo.py --run-id "apache/incubator-opendal_..."
  python monitoring_triage_demo.py --index 3
  python monitoring_triage_demo.py --list
  python monitoring_triage_demo.py --dry-run    # skip APA calls (risk score only)
  python monitoring_triage_demo.py --delay 0    # instant replay
  python monitoring_triage_demo.py --cases 5    # run on 5 different cases
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Windows UTF-8 output fix ─────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# ─────────────────────────────────────────────────────────────────────────────

from ci_stream_simulator import (
    DATA_FILE,
    load_cases,
    find_run,
    list_runs,
    stream_run_events,
    print_event,
    DEFAULT_DELAY,
)
from src.apa.ci_monitor import MonitorState, create_monitor, TRIAGE_THRESHOLD


# ─── APA integration ──────────────────────────────────────────────────────────

def call_apa(monitor: MonitorState, case: dict, label: str = "early") -> dict:
    """
    Build partial_raw_run from monitor state and invoke run_agent.
    Returns the agent result dict or an error stub.
    """
    try:
        from agent import run_agent
    except ImportError as e:
        return {"error": f"Could not import agent: {e}", "label": label}

    partial_run = monitor.build_partial_raw_run(case)

    print(f"\n  {'─'*60}")
    print(f"  ⚡ APA TRIAGE TRIGGERED [{label.upper()}]")
    print(f"     risk={monitor.failure_risk:.2f}  events={monitor.event_count}")
    print(f"     error_lines accumulated: {len(monitor.current_error_lines)}")
    print(f"     log chunks accumulated:  {len(monitor.seen_log_chunks)}")
    print(f"  {'─'*60}")

    t0 = time.time()
    try:
        result = run_agent(partial_run)
        elapsed = time.time() - t0
        result["_label"] = label
        result["_elapsed_sec"] = round(elapsed, 2)
        return result
    except Exception as exc:
        return {
            "error": str(exc),
            "label": label,
            "_label": label,
            "_elapsed_sec": round(time.time() - t0, 2),
        }


def print_triage_result(result: dict, title: str = "TRIAGE RESULT") -> None:
    """Pretty-print an agent result dict."""
    label = result.get("_label", "?")
    elapsed = result.get("_elapsed_sec", 0)

    print(f"\n  {'═'*60}")
    print(f"  {title}  [{label.upper()}]  ({elapsed:.1f}s)")
    print(f"  {'═'*60}")

    if "error" in result:
        print(f"  ⚠ Agent error: {result['error']}")
        return

    cl = result.get("classification", {})
    print(f"  category:    {cl.get('category', '?')}")
    print(f"  severity:    {cl.get('severity', '?')}")
    print(f"  confidence:  {cl.get('confidence', 0):.0%}")
    print(f"  reasoning:   {cl.get('reasoning', '')[:200]}")
    print(f"  fast_path:   {result.get('fast_path', False)}")
    print(f"  steps taken: {result.get('steps_taken', 0)}")
    print(f"  tools used:  {', '.join(result.get('tools_used', []) or []) or 'none'}")

    beliefs = result.get("beliefs") or {}
    if beliefs:
        top3 = sorted(beliefs.items(), key=lambda kv: -kv[1])[:3]
        for cat, prob in top3:
            bar = "█" * round(prob * 20) + "░" * (20 - round(prob * 20))
            print(f"  {cat:<28} [{bar}] {prob:.0%}")


def print_comparison(early: Optional[dict], final: dict, case: dict) -> None:
    """Print a side-by-side comparison of early vs final triage."""
    intake = case.get("intake", {})
    gt = case.get("ground_truth", {})

    print("\n" + "═" * 70)
    print("COMPARISON: EARLY TRIAGE vs FINAL TRIAGE vs GROUND TRUTH")
    print("═" * 70)

    rows = [
        ("Run ID",      intake.get("run_id", "?")[:50]),
        ("Repo",        intake.get("repo", "?")),
        ("Workflow",    intake.get("workflow", "?")),
        ("GT action",   gt.get("developer_action", "?")),
        ("GT verdict",  gt.get("match_verdict", "?")),
    ]
    for label, val in rows:
        print(f"  {label:<14} {val}")

    print()
    header = f"  {'Field':<18}  {'EARLY':<28}  {'FINAL':<28}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def _get(result: Optional[dict], key: str, default: str = "N/A") -> str:
        if result is None:
            return "N/A"
        if "error" in result:
            return f"ERROR: {result['error'][:20]}"
        cl = result.get("classification", {})
        return str(cl.get(key, default))[:26]

    fields = [
        ("category",   "category"),
        ("severity",   "severity"),
        ("confidence", "confidence"),
    ]
    for label, key in fields:
        early_val = _get(early, key)
        final_val = _get(final, key)
        if key == "confidence":
            try:
                early_val = f"{float(early_val):.0%}" if early_val != "N/A" else "N/A"
            except (ValueError, TypeError):
                pass
            try:
                final_val = f"{float(final_val):.0%}" if final_val != "N/A" else "N/A"
            except (ValueError, TypeError):
                pass
        match = "✓" if early_val == final_val else "✗"
        print(f"  {label:<18}  {early_val:<28}  {final_val:<28}  {match}")

    # Category agreement
    early_cat = _get(early, "category")
    final_cat = _get(final, "category")
    if early_cat == final_cat and early_cat not in ("N/A", "ERROR"):
        print("\n  ✅ Early triage matched final classification!")
    elif early_cat not in ("N/A", "ERROR"):
        print(f"\n  ⚠  Early vs final mismatch: {early_cat} ≠ {final_cat}")
    else:
        print("\n  (No early triage result to compare)")


# ─── single run demo ──────────────────────────────────────────────────────────

def run_demo(
    case: dict,
    delay: float = DEFAULT_DELAY,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full monitoring demo on a single case.

    Returns a summary dict with early_result, final_result, and monitor state.
    """
    intake = case.get("intake", {})
    run_id = intake.get("run_id", "?")

    print(f"\n{'═'*70}")
    print(f"  MONITORING: {run_id}")
    print(f"  repo={intake.get('repo')}  workflow={intake.get('workflow')}")
    print(f"  dry_run={'YES (no APA calls)' if dry_run else 'NO (full APA)'}")
    print(f"{'═'*70}\n")

    monitor = create_monitor(run_id=run_id)
    early_result: Optional[dict] = None
    final_result: Optional[dict] = None

    prev_risk = 0.0
    event_idx = 0

    for event in stream_run_events(case, delay=delay):
        event_idx += 1

        # Feed event to monitor
        fired_signals = monitor.process_event(event)

        # Print event
        if verbose:
            print_event(event_idx, event)

        # Print risk update if it changed meaningfully
        risk_changed = abs(monitor.failure_risk - prev_risk) >= 0.01
        if risk_changed and verbose:
            # Colour: green = low, yellow = medium, red = high
            risk = monitor.failure_risk
            if risk >= 0.75:
                colour = "\033[91m"  # red
            elif risk >= 0.45:
                colour = "\033[93m"  # yellow
            else:
                colour = "\033[92m"  # green
            reset = "\033[0m"
            signals_str = ", ".join(fired_signals) if fired_signals else "—"
            print(
                f"       {colour}risk → {risk:.2f}  {monitor.risk_bar}{reset}"
                f"  signals: {signals_str}"
            )
            prev_risk = monitor.failure_risk

        # ── Early triage trigger ─────────────────────────────────────────────
        if monitor.should_triage and early_result is None:
            monitor.triage_triggered = True

            if dry_run:
                print(f"\n  [DRY-RUN] Would trigger APA here. risk={monitor.failure_risk:.2f}")
                early_result = {"_label": "early", "dry_run": True, "risk": monitor.failure_risk}
            else:
                early_result = call_apa(monitor, case, label="early")
                monitor.triage_result = early_result
                # Update beliefs from early triage
                if "beliefs" in (early_result or {}):
                    monitor.current_beliefs = early_result["beliefs"]
                print_triage_result(early_result, title="EARLY TRIAGE RESULT")

    # ── Final triage (on confirmed failure, full evidence) ───────────────────
    if monitor.confirmed_failed:
        print(f"\n\n{'─'*70}")
        print(f"  Run confirmed FAILED. Running final APA triage...")
        print(f"{'─'*70}")

        if dry_run:
            final_result = {"_label": "final", "dry_run": True}
        else:
            # For final triage, mark as confirmed failure
            final_monitor = MonitorState(
                run_id=monitor.run_id,
                repo=monitor.repo,
                workflow=monitor.workflow,
                branch=monitor.branch,
                commit_sha=monitor.commit_sha,
                commit_title=monitor.commit_title,
                event_trigger=monitor.event_trigger,
                failure_risk=monitor.failure_risk,
                confirmed_failed=True,
                seen_log_chunks=list(monitor.seen_log_chunks),
                current_error_lines=list(monitor.current_error_lines),
                jobs_seen=list(monitor.jobs_seen),
                steps_seen=list(monitor.steps_seen),
                failed_steps=list(monitor.failed_steps),
                error_events=list(monitor.error_events),
                signals_fired=set(monitor.signals_fired),
                risk_log=list(monitor.risk_log),
                event_count=monitor.event_count,
            )
            final_result = call_apa(final_monitor, case, label="final")
            print_triage_result(final_result, title="FINAL TRIAGE RESULT")

    # ── Comparison ───────────────────────────────────────────────────────────
    print_comparison(early_result, final_result or {}, case)

    # ── Risk score trace ─────────────────────────────────────────────────────
    if verbose and monitor.risk_log:
        print(f"\n  Risk signal trace:")
        for entry in monitor.risk_log:
            print(
                f"    event#{entry['event_count']:<4}  "
                f"{entry['signal']:<30}  "
                f"+{entry['delta']:.2f}  "
                f"→  {entry['new_risk']:.2f}"
            )

    return {
        "run_id": run_id,
        "early_result": early_result,
        "final_result": final_result,
        "monitor_state": {
            "failure_risk": monitor.failure_risk,
            "triage_triggered": monitor.triage_triggered,
            "confirmed_failed": monitor.confirmed_failed,
            "event_count": monitor.event_count,
            "signals_fired": list(monitor.signals_fired),
            "error_lines_seen": len(monitor.current_error_lines),
            "log_chunks_seen": len(monitor.seen_log_chunks),
        },
        "ground_truth": case.get("ground_truth", {}),
    }


# ─── batch mode ───────────────────────────────────────────────────────────────

def run_batch(
    cases: List[dict],
    n: int = 5,
    delay: float = 0.0,
    dry_run: bool = True,
) -> None:
    """
    Run the monitoring demo on N cases silently (dry_run recommended for batch).
    Prints a summary table at the end.
    """
    results = []
    selected = [c for c in cases if c.get("intake", {}).get("conclusion") == "failure"][:n]

    print(f"Running batch monitoring demo on {len(selected)} cases...")
    print(f"(dry_run={'YES' if dry_run else 'NO'})\n")

    for i, case in enumerate(selected):
        rid = case.get("intake", {}).get("run_id", "?")
        print(f"[{i+1}/{len(selected)}] {rid[:60]}...", end="", flush=True)
        summary = run_demo(case, delay=delay, dry_run=dry_run, verbose=False)
        ms = summary["monitor_state"]
        print(
            f"  risk={ms['failure_risk']:.2f}"
            f"  triage={'YES' if ms['triage_triggered'] else 'NO'}"
            f"  errors={ms['error_lines_seen']}"
        )
        results.append(summary)

    # Summary table
    print(f"\n{'═'*70}")
    print("BATCH SUMMARY")
    print(f"{'═'*70}")
    triggered = sum(1 for r in results if r["monitor_state"]["triage_triggered"])
    print(f"  Cases run:         {len(results)}")
    print(f"  Triage triggered:  {triggered}/{len(results)}")
    avg_risk = sum(r["monitor_state"]["failure_risk"] for r in results) / len(results) if results else 0
    print(f"  Avg final risk:    {avg_risk:.2f}")
    print()

    print(f"  {'#':<3}  {'Risk':>5}  {'Triage?':<8}  {'Run ID'}")
    print("  " + "-" * 65)
    for i, r in enumerate(results):
        ms = r["monitor_state"]
        trig = "YES" if ms["triage_triggered"] else "no"
        rid = r["run_id"][:50]
        print(f"  {i+1:<3}  {ms['failure_risk']:>5.2f}  {trig:<8}  {rid}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous CI monitoring + APA triage demo."
    )
    parser.add_argument("--run-id", metavar="RUN_ID", help="Specific run_id to demo.")
    parser.add_argument("--index", type=int, metavar="N", help="Pick case by 0-based index.")
    parser.add_argument("--list", action="store_true", help="List available failed runs.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        metavar="SEC", help=f"Pause between events (default {DEFAULT_DELAY}s). 0=instant.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip APA calls — show risk score only.")
    parser.add_argument("--cases", type=int, metavar="N",
                        help="Batch mode: run N failed cases (dry-run by default).")
    parser.add_argument("--data", default=str(DATA_FILE), metavar="PATH",
                        help="Path to targeted_cases.json.")
    parser.add_argument("--no-verbose", action="store_true",
                        help="Suppress per-event output (useful for batch).")
    args = parser.parse_args()

    cases = load_cases(Path(args.data))
    failed_cases = [c for c in cases if c.get("intake", {}).get("conclusion") == "failure"]

    if args.list:
        list_runs(failed_cases)
        return

    if args.cases:
        run_batch(
            cases,
            n=args.cases,
            delay=args.delay,
            dry_run=args.dry_run or True,  # default batch to dry_run
        )
        return

    # Single-run mode
    if args.run_id:
        case = find_run(cases, args.run_id)
        if case is None:
            print(f"ERROR: run_id not found: {args.run_id}", file=sys.stderr)
            sys.exit(1)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(failed_cases):
            print(f"ERROR: index {args.index} out of range (0–{len(failed_cases)-1})", file=sys.stderr)
            sys.exit(1)
        case = failed_cases[args.index]
    else:
        # Default: pick the second failed run (first one is often a quick one)
        if not failed_cases:
            print("ERROR: No failed runs found.", file=sys.stderr)
            sys.exit(1)
        case = failed_cases[1] if len(failed_cases) > 1 else failed_cases[0]
        rid = case.get("intake", {}).get("run_id", "?")
        print(f"(No --run-id specified, using: {rid})\n")

    run_demo(
        case,
        delay=args.delay,
        dry_run=args.dry_run,
        verbose=not args.no_verbose,
    )


if __name__ == "__main__":
    main()
