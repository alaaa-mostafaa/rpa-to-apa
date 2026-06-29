# eval_dual_tracker.py
# Run the dual-mode Bayesian tracker (RPA vs APA) on multiple cases
# and compare which mode gets closer to the ground truth.

import gzip
import json
import os
from pathlib import Path
from dataclasses import asdict

from dotenv import load_dotenv
from openai import OpenAI
from src.apa.llm_config import make_client

from src.apa.intake_parser import intake
from src.apa.log_extractor import extract_log_excerpt
from src.apa.bayesian_tracker_dual import DualTracker
from src.apa.bayesian_tracker import print_beliefs

load_dotenv()

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"
OUTPUT_PATH = Path("/home/guc_alaa/dual_tracker_eval.json")

# Use the same cases from the honest evaluation so we can compare
HONEST_EVAL_PATH = Path("/home/guc_alaa/honest_eval_results.json")


def tarball_path(raw: dict) -> str:
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/"):] if p.startswith("/data/") else p


def get_error_lines_from_excerpts(raw: dict, event) -> list:
    """Extract sample error lines for the tracker."""
    tarball = tarball_path(raw)
    error_lines = []
    for fs in event.failed_steps[:3]:  # limit to 3 steps to save API calls
        ex = extract_log_excerpt(
            zip_path=ZIP_PATH,
            tarball_name=tarball,
            job_file=fs.job_file,
            step_label=fs.step_label or "",
        )
        for ml in ex.error_marker_lines[:2]:
            cleaned = ml.strip()[:120]
            if cleaned and cleaned not in error_lines:
                error_lines.append(cleaned)
        for w in ex.error_windows:
            for ln in w[-5:]:
                ln_clean = ln.strip()
                if any(kw in ln_clean.lower() for kw in
                       ("error", "failed", "fatal", "exception",
                        "not found", "denied", "timeout")):
                    if ln_clean not in error_lines and len(error_lines) < 5:
                        error_lines.append(ln_clean[:120])
    return error_lines


def run_tracker(mode: str, event, error_lines: list, client=None) -> dict:
    """Run one mode of the tracker and return results."""
    tracker = DualTracker(mode=mode, client=client)

    tracker.observe_branch(event.is_protected_branch, event.branch)
    tracker.observe_jobs(event.failed_jobs_count, event.n_jobs)
    if error_lines:
        tracker.observe_errors(error_lines)
    tracker.observe_commit(event.commit_message or event.commit_title)
    tracker.observe_detection(event.failure_detection)

    return tracker.result()


def main():
    try:
        client = make_client()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    # Load the honest eval results for ground truth comparison
    if not HONEST_EVAL_PATH.exists():
        print(f"ERROR: {HONEST_EVAL_PATH} not found. Run honest_evaluation.py first.")
        return

    honest_results = json.load(open(HONEST_EVAL_PATH))
    # Build a lookup by run_id
    honest_by_id = {r["intake"]["run_id"]: r for r in honest_results}

    # Get the run IDs we want to evaluate
    run_ids = [r["intake"]["run_id"] for r in honest_results]

    # Find the raw runs in the dataset
    print(f"Finding {len(run_ids)} runs in runs.json.gz...")
    wanted = set(run_ids)
    raw_runs = {}
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = run.get("_id")
            if rid in wanted:
                raw_runs[rid] = run
                if len(raw_runs) == len(wanted):
                    break
    print(f"Found {len(raw_runs)}/{len(run_ids)} runs.\n")

    # Run both trackers on each case
    results = []
    rpa_score = {"correct": 0, "total": 0}
    apa_score = {"correct": 0, "total": 0}

    for i, run_id in enumerate(run_ids[:30], 1):  # limit to 30 to save API costs
        raw = raw_runs.get(run_id)
        if not raw:
            continue

        event = intake(raw)
        honest = honest_by_id.get(run_id, {})
        honest_category = honest.get("classification", {}).get("category", "?")
        gt_action = honest.get("ground_truth", {}).get("developer_action", "?")
        gt_verdict = honest.get("ground_truth", {}).get("match_verdict", "?")

        print(f"[{i}] {event.repo} — {event.commit_title[:40]}")

        # Get error lines
        error_lines = get_error_lines_from_excerpts(raw, event)

        # RPA mode (instant)
        rpa_result = run_tracker("rpa", event, error_lines)

        # APA mode (API calls)
        apa_result = run_tracker("apa", event, error_lines, client=client)

        # Compare against the honest eval's LLM classification
        # (using it as a reference, not as absolute truth)
        rpa_matches = rpa_result["category"] == honest_category
        apa_matches = apa_result["category"] == honest_category

        if honest_category != "?":
            rpa_score["total"] += 1
            apa_score["total"] += 1
            if rpa_matches:
                rpa_score["correct"] += 1
            if apa_matches:
                apa_score["correct"] += 1

        # Determine agreement
        if rpa_result["category"] == apa_result["category"]:
            agreement = "AGREE"
        else:
            agreement = "DISAGREE"

        print(f"  RPA: {rpa_result['category']:<25} (p={rpa_result['probability']:.2f}, conf={rpa_result['confidence']:.0%})")
        print(f"  APA: {apa_result['category']:<25} (p={apa_result['probability']:.2f}, conf={apa_result['confidence']:.0%})")
        print(f"  LLM: {honest_category:<25} GT: {gt_action}")
        print(f"  {agreement} | RPA={'✓' if rpa_matches else '✗'} | APA={'✓' if apa_matches else '✗'}")
        print()

        results.append({
            "run_id": run_id,
            "repo": event.repo,
            "commit": event.commit_title[:60],
            "rpa": rpa_result,
            "apa": apa_result,
            "llm_category": honest_category,
            "ground_truth_action": gt_action,
            "ground_truth_verdict": gt_verdict,
            "agreement": agreement,
            "rpa_matches_llm": rpa_matches,
            "apa_matches_llm": apa_matches,
        })

        # Save incrementally
        OUTPUT_PATH.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n = len(results)
    agree = sum(1 for r in results if r["agreement"] == "AGREE")
    disagree = n - agree
    print(f"\n  Cases evaluated:   {n}")
    print(f"  RPA & APA agree:   {agree} ({agree*100//n if n else 0}%)")
    print(f"  RPA & APA disagree: {disagree} ({disagree*100//n if n else 0}%)")

    if rpa_score["total"] > 0:
        print(f"\n  Agreement with LLM classifier (reference):")
        print(f"    RPA: {rpa_score['correct']}/{rpa_score['total']} "
              f"({rpa_score['correct']*100//rpa_score['total']}%)")
        print(f"    APA: {apa_score['correct']}/{apa_score['total']} "
              f"({apa_score['correct']*100//apa_score['total']}%)")

    # Category distribution comparison
    rpa_cats = [r["rpa"]["category"] for r in results]
    apa_cats = [r["apa"]["category"] for r in results]
    print(f"\n  Category distribution:")
    print(f"    {'Category':<25} {'RPA':>5} {'APA':>5} {'LLM':>5}")
    all_cats = sorted(set(rpa_cats + apa_cats + [r["llm_category"] for r in results]))
    for cat in all_cats:
        r_count = rpa_cats.count(cat)
        a_count = apa_cats.count(cat)
        l_count = sum(1 for r in results if r["llm_category"] == cat)
        print(f"    {cat:<25} {r_count:>5} {a_count:>5} {l_count:>5}")

    # Confidence comparison
    rpa_conf = [r["rpa"]["confidence"] for r in results]
    apa_conf = [r["apa"]["confidence"] for r in results]
    print(f"\n  Average confidence:")
    print(f"    RPA: {sum(rpa_conf)/len(rpa_conf):.1%}")
    print(f"    APA: {sum(apa_conf)/len(apa_conf):.1%}")

    # Disagreement details
    print(f"\n  DISAGREEMENTS:")
    for r in results:
        if r["agreement"] == "DISAGREE":
            print(f"    {r['repo'][:25]:<27} "
                  f"RPA={r['rpa']['category']:<22} "
                  f"APA={r['apa']['category']:<22} "
                  f"LLM={r['llm_category']}")

    print(f"\n  Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()