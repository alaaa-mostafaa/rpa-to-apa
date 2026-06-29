"""Re-judge the 250 scorable cases from benchmark_final_eval.json using GPT-4o.

APA agent predictions are already saved — this only re-runs the judge.
Output is written to data/benchmark_gpt4o_judged.json (originals untouched).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

os.environ["JUDGE_MODEL"] = "gpt-4o"

from openai import OpenAI
from evals.evaluation_judge import llm_judge, build_case_summary

INPUT_PATH  = Path("data/benchmark_final_eval.json")
OUTPUT_PATH = Path("data/benchmark_gpt4o_judged.json")

SCORABLE_VERDICTS = {"CORRECT", "PARTIAL", "WRONG"}
N_TARGET = 250


class MinimalEvent:
    def __init__(self, r):
        self.repo               = r.get("repo", "")
        self.workflow           = r.get("workflow", "") or ""
        self.branch             = r.get("branch", "") or ""
        self.is_protected_branch = False
        self.commit_sha         = r.get("run_id", "")
        self.commit_title       = r.get("commit", "")
        self.event              = ""
        self.conclusion         = "failure"
        self.failed_jobs_count  = 1
        self.n_jobs             = 1
        self.failure_detection  = ""
        self.run_id             = r.get("run_id", "")


class StoredGT:
    def __init__(self, gt_stored):
        self.developer_action           = gt_stored.get("action", "UNKNOWN")
        self.classification_method      = gt_stored.get("method", "stored")
        self.developer_action_reasoning = gt_stored.get("reasoning", "")
        self.follow_up_commits          = []
        self.signals                    = {}


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    # Pick first N_TARGET scorable cases (same 250 used in the thesis)
    scorable = [
        r for r in data
        if r.get("rpa", {}).get("judge", {}).get("verdict") in SCORABLE_VERDICTS
        and r.get("apa_eig", {}).get("judge", {}).get("verdict") in SCORABLE_VERDICTS
    ]
    targets = scorable[:N_TARGET]
    print(f"Total scorable: {len(scorable)}  |  Re-judging first {len(targets)} with GPT-4o")

    # Load existing output if present (resume support)
    if OUTPUT_PATH.exists():
        out_data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        done_ids = {r["run_id"] for r in out_data}
        print(f"Resuming — {len(done_ids)} already done.")
    else:
        out_data = []
        done_ids = set()

    client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

    for idx, r in enumerate(targets, 1):
        run_id = r.get("run_id", "")
        if run_id in done_ids:
            continue

        repo = r.get("repo", "?")
        print(f"[{idx}/{len(targets)}] {repo} ...", end=" ", flush=True)

        try:
            event        = MinimalEvent(r)
            gt           = StoredGT(r.get("ground_truth", {}))
            error_lines  = r.get("error_lines", [])
            case_summary = build_case_summary(event, error_lines)

            rpa_judge = llm_judge(case_summary, r["rpa"]["prediction"], gt, client)
            apa_judge = llm_judge(case_summary, r["apa_eig"]["prediction"], gt, client)

            out_rec = {
                "run_id":       run_id,
                "repo":         repo,
                "commit":       r.get("commit", ""),
                "ground_truth": r.get("ground_truth", {}),
                "error_lines":  error_lines,
                "rpa": {
                    "prediction": r["rpa"]["prediction"],
                    "judge":      rpa_judge,
                },
                "apa_eig": {
                    "prediction": r["apa_eig"]["prediction"],
                    "judge":      apa_judge,
                },
            }
            out_data.append(out_rec)
            done_ids.add(run_id)

            print(f"RPA={rpa_judge['verdict']}  APA={apa_judge['verdict']}")

            # Save after every case
            OUTPUT_PATH.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}")
    print(f"Judged {len(out_data)} cases with GPT-4o")
    rpa_correct  = sum(1 for r in out_data if r["rpa"]["judge"]["verdict"] == "CORRECT")
    apa_correct  = sum(1 for r in out_data if r["apa_eig"]["judge"]["verdict"] == "CORRECT")
    rpa_partial  = sum(1 for r in out_data if r["rpa"]["judge"]["verdict"] in ("CORRECT","PARTIAL"))
    apa_partial  = sum(1 for r in out_data if r["apa_eig"]["judge"]["verdict"] in ("CORRECT","PARTIAL"))
    n = len(out_data)
    print(f"{'':30} {'Strict':>8} {'Strict%':>9} {'+Partial':>10} {'+Partial%':>11}")
    print(f"{'L1 — RPA':30} {rpa_correct:>8} {rpa_correct/n*100:>8.1f}% {rpa_partial:>10} {rpa_partial/n*100:>10.1f}%")
    print(f"{'L2 — APA':30} {apa_correct:>8} {apa_correct/n*100:>8.1f}% {apa_partial:>10} {apa_partial/n*100:>10.1f}%")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
