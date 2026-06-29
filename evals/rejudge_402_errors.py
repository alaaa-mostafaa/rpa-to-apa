"""Re-judge cases that failed due to 402 Insufficient Balance errors.
The APA agent predictions are already saved — we just need to re-run the judge.
"""
import json
import os
import sys
from pathlib import Path

# Set up env before importing APA modules
os.environ.setdefault("LLM_PROVIDER", "deepseek")
os.environ.setdefault("CI_AGENT_MODEL", "deepseek-chat")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.apa.intake_parser import RunEvent
from archive.ground_truth_scraper import scrape_ground_truth
from evals.evaluation_judge import llm_judge, build_case_summary
from src.apa.llm_config import make_client

OUTPUT_PATH = Path("data/benchmark_1000_eig_vs_rpa.json")

def is_402_error(verdict_dict: dict) -> bool:
    reasoning = verdict_dict.get("reasoning", "")
    return (
        verdict_dict.get("verdict") == "EVALUATION_ERROR"
        and ("402" in reasoning or "Insufficient Balance" in reasoning)
    )

def main():
    data = json.loads(OUTPUT_PATH.read_text())
    
    # Find cases with 402 errors in either verdict
    targets = [
        (i, r) for i, r in enumerate(data)
        if is_402_error(r.get("rpa", {}).get("judge", {}))
        or is_402_error(r.get("apa_eig", {}).get("judge", {}))
    ]
    
    print(f"Found {len(targets)} cases to re-judge (402 billing errors).")
    if not targets:
        print("Nothing to do.")
        return

    client = make_client()
    fixed = 0

    for idx, (i, r) in enumerate(targets, 1):
        repo = r.get("repo", "?")
        print(f"\n[{idx}/{len(targets)}] Re-judging {repo}...")

        # Rebuild the RunEvent to get ground truth and case summary
        try:
            # We stored run_id and repo, but need the original event dict.
            # Reconstruct a minimal RunEvent from stored data.
            run_id = r.get("run_id", "")
            commit = r.get("commit", "")

            # We need to re-scrape ground truth (it's not stored in full)
            # Find the original raw data isn't available here, so we use
            # what's stored in the result to build a minimal fake event.
            class MinimalEvent:
                repo = r.get("repo", "")
                workflow = ""
                branch = ""
                is_protected_branch = False
                commit_sha = ""
                commit_title = r.get("commit", "")
                event = ""
                conclusion = "failure"
                failed_jobs_count = 1
                n_jobs = 1
                failure_detection = ""
                run_id = r.get("run_id", "")

            gt_stored = r.get("ground_truth", {})

            # Use stored ground truth info to build a GroundTruth-like object
            class StoredGT:
                developer_action = gt_stored.get("action", "UNKNOWN")
                classification_method = gt_stored.get("method", "stored")
                developer_action_reasoning = gt_stored.get("reasoning", "")
                follow_up_commits = []
                signals = {}

            event = MinimalEvent()
            gt = StoredGT()

            case_summary = build_case_summary(event, [])

            # Re-judge only the sides that had 402 errors
            rpa_judge = r["rpa"]["judge"]
            apa_judge = r["apa_eig"]["judge"]

            if is_402_error(rpa_judge):
                print(f"  Re-judging RPA...")
                rpa_judge = llm_judge(case_summary, r["rpa"]["prediction"], gt, client)
                data[i]["rpa"]["judge"] = rpa_judge
                print(f"  RPA verdict: {rpa_judge['verdict']}")

            if is_402_error(apa_judge):
                print(f"  Re-judging APA...")
                apa_judge = llm_judge(case_summary, r["apa_eig"]["prediction"], gt, client)
                data[i]["apa_eig"]["judge"] = apa_judge
                print(f"  APA verdict: {apa_judge['verdict']}")

            fixed += 1
            # Save after each case in case of interruption
            OUTPUT_PATH.write_text(json.dumps(data, indent=2))

        except Exception as e:
            print(f"  Error on {repo}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Done. Re-judged {fixed}/{len(targets)} cases. ===")

    # Quick summary
    scorable = sum(
        1 for r in data
        if r.get("rpa", {}).get("judge", {}).get("verdict") not in ("NOT_SCORABLE", "EVALUATION_ERROR", "")
        and r.get("apa_eig", {}).get("judge", {}).get("verdict") not in ("NOT_SCORABLE", "EVALUATION_ERROR", "")
    )
    apa_correct = sum(1 for r in data if r.get("apa_eig", {}).get("judge", {}).get("verdict") == "CORRECT")
    rpa_correct = sum(1 for r in data if r.get("rpa", {}).get("judge", {}).get("verdict") == "CORRECT")
    print(f"Updated scorable cases: {scorable}")
    print(f"APA correct: {apa_correct}  |  RPA correct: {rpa_correct}")
    if scorable:
        print(f"APA accuracy: {apa_correct/scorable*100:.1f}%  |  RPA accuracy: {rpa_correct/scorable*100:.1f}%")

if __name__ == "__main__":
    main()
