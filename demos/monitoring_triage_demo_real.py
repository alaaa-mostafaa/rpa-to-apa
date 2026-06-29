"""
monitoring_triage_demo_real.py
─────────────────────────────────────────────────────────────────────────────
End-to-end monitoring demo using the REAL full_log.txt files from the
50-case balanced dataset (25 failure, 25 success).

Every log line the monitor sees is a real GitHub Actions log line.
No synthetic text. No fake compilation output.

Usage
─────
  python monitoring_triage_demo_real.py --list
  python monitoring_triage_demo_real.py --case 005        # specific case
  python monitoring_triage_demo_real.py --index 0         # first failure
  python monitoring_triage_demo_real.py --index 25        # first success
  python monitoring_triage_demo_real.py --all             # run all 50
  python monitoring_triage_demo_real.py --all --failures-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ci_stream_simulator_real import (
    REAL_LOGS_DIR,
    load_real_case_dirs,
    stream_real_events,
    print_event,
    find_case,
    list_cases,
)
# Add project root to sys.path so we can import src.apa.*
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.apa.ci_monitor import MonitorState, TRIAGE_THRESHOLD

TRIAGE_THRESHOLD_USED = TRIAGE_THRESHOLD


# ─── demo for one case ────────────────────────────────────────────────────────

def run_real_demo(
    case: dict,
    verbose: bool = True,
    delay: float = 0.0,
) -> dict:
    """
    Stream a real log file through the monitor and report what happened.
    Returns a summary dict.
    """
    run_id = case.get("run_id", case["_case_name"])
    label = case["_label"]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  REAL MONITORING: {case['_case_name']}")
        print(f"  run_id = {run_id}")
        print(f"  truth  = {label.upper()}")
        print(f"{'='*70}\n")

    monitor = MonitorState(run_id=run_id)
    prev_risk = 0.0
    triage_event_idx: Optional[int] = None
    total_events = 0
    first_error_event_idx: Optional[int] = None

    for event in stream_real_events(case, delay=delay):
        total_events += 1

        fired = monitor.process_event(event)

        if verbose:
            print_event(total_events, event)

        # Track first real ##[error] event
        if event["type"] == "error_seen" and first_error_event_idx is None:
            first_error_event_idx = total_events

        # Risk update display
        if fired and verbose:
            risk = monitor.failure_risk
            colour = "\033[91m" if risk >= 0.75 else "\033[93m" if risk >= 0.45 else "\033[92m"
            reset = "\033[0m"
            print(f"       {colour}risk -> {risk:.2f}  {monitor.risk_bar}{reset}"
                  f"  signals: {', '.join(fired)}")
            prev_risk = risk

        # Triage trigger
        if monitor.should_triage and triage_event_idx is None:
            monitor.triage_triggered = True
            triage_event_idx = total_events
            if verbose:
                print(f"\n  *** TRIAGE TRIGGERED at event #{total_events} ***")
                print(f"      risk={monitor.failure_risk:.2f}  truth={label.upper()}")
                print(f"      error_lines_seen={len(monitor.current_error_lines)}")
                if first_error_event_idx:
                    print(f"      first ##[error] was at event #{first_error_event_idx}")
                    if triage_event_idx < first_error_event_idx:
                        print(f"      --> PREDICTED BEFORE ##[error] by "
                              f"{first_error_event_idx - triage_event_idx} events!")
                print()

    # Outcome classification
    predicted = "failure" if monitor.failure_risk >= TRIAGE_THRESHOLD_USED else "success"
    correct = predicted == label

    if verbose:
        print(f"\n{'='*70}")
        print(f"  RESULT")
        print(f"  Ground truth:   {label.upper()}")
        print(f"  Predicted:      {predicted.upper()}  (risk={monitor.failure_risk:.2f})")
        print(f"  Correct:        {'YES' if correct else 'NO'}")
        print(f"  Triage triggered: {'YES at event #' + str(triage_event_idx) if triage_event_idx else 'NO'}")
        print(f"  Total events:   {total_events}")
        print()
        if monitor.risk_log:
            print("  Risk trace:")
            for entry in monitor.risk_log:
                print(f"    event#{entry['event_count']:<4}  "
                      f"{entry['signal']:<30}  +{entry['delta']:.2f}  -> {entry['new_risk']:.2f}")
        print(f"{'='*70}")

    return {
        "case": case["_case_name"],
        "run_id": run_id,
        "truth": label,
        "predicted": predicted,
        "correct": correct,
        "final_risk": round(monitor.failure_risk, 3),
        "triage_triggered": triage_event_idx is not None,
        "triage_event_idx": triage_event_idx,
        "first_error_event_idx": first_error_event_idx,
        "total_events": total_events,
        "signals_fired": list(monitor.signals_fired),
        "error_lines_seen": len(monitor.current_error_lines),
        "predicted_before_error": (
            triage_event_idx is not None
            and first_error_event_idx is not None
            and triage_event_idx < first_error_event_idx
        ),
    }


# ─── batch mode ───────────────────────────────────────────────────────────────

def run_all(cases, verbose: bool = False, delay: float = 0.0) -> None:
    results = []
    for i, case in enumerate(cases):
        print(f"[{i+1:>3}/{len(cases)}] {case['_case_name'][:55]}...", end="", flush=True)
        r = run_real_demo(case, verbose=False, delay=delay)
        results.append(r)
        status = "OK " if r["correct"] else "ERR"
        trig = f"triage@{r['triage_event_idx']}" if r["triage_triggered"] else "no triage"
        print(f"  {status}  risk={r['final_risk']:.2f}  {trig}")

    # ── Summary table ──────────────────────────────────────────────────────────
    failures = [r for r in results if r["truth"] == "failure"]
    successes = [r for r in results if r["truth"] == "success"]

    tp = sum(1 for r in failures if r["correct"])    # failure correctly flagged
    tn = sum(1 for r in successes if r["correct"])   # success correctly left alone
    fp = sum(1 for r in successes if not r["correct"])  # success wrongly flagged
    fn = sum(1 for r in failures if not r["correct"])   # failure missed

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # "Predicted before ##[error]" — truly early prediction
    early_preds = sum(1 for r in failures if r.get("predicted_before_error"))

    print(f"\n{'='*70}")
    print("REAL LOG MONITORING EVALUATION")
    print(f"{'='*70}")
    print(f"  Cases: {len(results)} total ({len(failures)} failure, {len(successes)} success)")
    print()
    print(f"  True  Positives (failure correctly flagged):  {tp}/{len(failures)}")
    print(f"  True  Negatives (success correctly left):     {tn}/{len(successes)}")
    print(f"  False Positives (success wrongly flagged):    {fp}/{len(successes)}")
    print(f"  False Negatives (failure missed):             {fn}/{len(failures)}")
    print()
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 score:  {f1:.2%}")
    print()
    print(f"  Early predictions (triage BEFORE ##[error] appeared): {early_preds}/{len(failures)}")
    print()

    # Risk distribution
    fail_risks = [r["final_risk"] for r in failures]
    succ_risks = [r["final_risk"] for r in successes]
    avg_fail = sum(fail_risks) / len(fail_risks) if fail_risks else 0
    avg_succ = sum(succ_risks) / len(succ_risks) if succ_risks else 0
    print(f"  Avg risk — failure cases: {avg_fail:.3f}")
    print(f"  Avg risk — success cases: {avg_succ:.3f}")
    print(f"  Risk separation: {avg_fail - avg_succ:.3f}")

    # Save results
    out_path = Path("comparison_results/balanced_50_full_logs_20260516/monitor_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "early_predictions": early_preds,
                "avg_risk_failure": round(avg_fail, 4),
                "avg_risk_success": round(avg_succ, 4),
            },
            "cases": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor demo using REAL CI logs — no synthetic data."
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", metavar="NNN")
    parser.add_argument("--index", type=int, metavar="N")
    parser.add_argument("--all", action="store_true", help="Run all 50 cases.")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--successes-only", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true", help="Show per-event output in --all mode.")
    args = parser.parse_args()

    cases = load_real_case_dirs(REAL_LOGS_DIR)
    if args.failures_only:
        cases = [c for c in cases if c["_label"] == "failure"]
    elif args.successes_only:
        cases = [c for c in cases if c["_label"] == "success"]

    if args.list:
        list_cases(cases)
        return

    if args.all:
        run_all(cases, verbose=args.verbose, delay=args.delay)
        return

    if args.case:
        case = find_case(cases, args.case)
        if not case:
            print(f"ERROR: case '{args.case}' not found", file=sys.stderr)
            sys.exit(1)
    elif args.index is not None:
        case = cases[args.index]
    else:
        case = cases[0]
        print(f"(No case specified, defaulting to: {case['_case_name']})\n")

    run_real_demo(case, verbose=True, delay=args.delay)


if __name__ == "__main__":
    main()
