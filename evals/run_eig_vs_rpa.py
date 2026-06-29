import gzip
import json
import os
import sys
import traceback
from pathlib import Path
from collections import Counter
from dataclasses import asdict
from datetime import datetime

from dotenv import load_dotenv

# Set up environment for the run BEFORE importing APA modules
os.environ["CI_AGENT_PLANNER_MODE"] = "hybrid"  # hybrid = EIG + LLM
os.environ["LLM_PROVIDER"] = "deepseek"
os.environ["CI_AGENT_MODEL"] = "deepseek-reasoner"  # Reasoner model
os.environ["JUDGE_MODEL"] = "deepseek-reasoner"

# Use Gemini models
MODEL_EASY = "deepseek-chat"
MODEL_HARD = "deepseek-reasoner"
CONFIDENCE_THRESHOLD = 0.70   # RPA confidence above this -> EASY tier
ENTROPY_THRESHOLD    = 1.50   # RPA entropy   below this -> EASY tier

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.apa.intake_parser import intake
from src.apa.agent import run_agent
from archive.ground_truth_scraper import scrape_ground_truth, quick_gt_preflight
from evals.evaluation_judge import llm_judge, build_case_summary
from src.apa.bayesian_tracker_dual import DualTracker
from src.apa.llm_config import make_client

load_dotenv()

RUNS_PATH = Path("data/intake_logs_5000.jsonl.gz")
OUTPUT_PATH = Path("data/benchmark_final_eval.json")  # new file — preserves old benchmark
QUALITATIVE_PATH = Path("data/qualitative_analysis_final.json")
METRICS_PATH = Path("data/metrics_summary_final.json")

# Repos already used in previous benchmark runs — skip these for a fresh sample
SEEN_REPOS_FROM_OLD_BENCHMARK = Path("data/benchmark_1000_eig_vs_rpa.json")

N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SCORABLE_PREFLIGHT = {"CODE_FIX", "WORKFLOW_FIX", "PIN_VERSION", "REVERT", "CODE_CHANGE", "PR_MERGED"}

def run_case(raw: dict, client) -> dict:
    from src.apa.intake_parser import RunEvent
    event_dict = raw.get("intake", {})
    
    # Fill in defaults for missing fields that were added to RunEvent later
    defaults = {
        "source": "github",
        "run_number": 0, "attempt": 1, "actor": "", 
        "commit_message": event_dict.get("commit_title", ""), 
        "commit_author": "", "started_at": "", "duration_sec": 0
    }
    kwargs = {**defaults, **event_dict}
    
    event = RunEvent(**{k: v for k, v in kwargs.items() if k in RunEvent.__dataclass_fields__})
    
    # 1. Run RPA Tracker (No LLM) — fast, always runs
    tracker = DualTracker(mode="rpa", client=None)
    tracker.observe_branch(event.is_protected_branch, event.branch)
    tracker.observe_jobs(event.failed_jobs_count, event.n_jobs)
    tracker.observe_commit(event.commit_message or event.commit_title)
    tracker.observe_detection(event.failure_detection)
    rpa_result = tracker.result()

    # Get pre-extracted logs
    extraction = raw.get("extraction", {})
    excerpts = extraction.get("log_excerpts", [])
    log_texts = [ex.get("text", "") for ex in excerpts if isinstance(ex, dict) and "text" in ex]
    error_lines = extraction.get("sample_error_lines", [])

    # 2. Scrape Ground Truth FIRST — uses GitHub API (no LLM for most cases).
    #    This is cheap and lets us bail before expensive APA + judge LLM calls.
    NOT_SCORABLE_ACTIONS = {
        "UNRELATED", "PR_CLOSED_UNMERGED", "PR_MERGED",
        "EVALUATION_ERROR", "UNKNOWN", "NO_IMMEDIATE_RESPONSE",
    }
    from archive.ground_truth_scraper import scrape_ground_truth
    gt = scrape_ground_truth(event)
    dev_action = getattr(gt, "developer_action", "UNKNOWN")

    # Build a cheap RPA-only result so we can save something even when skipping
    case_summary = build_case_summary(event, [])

    if dev_action in NOT_SCORABLE_ACTIONS:
        # Skip all LLM calls — this case is not scorable
        print(f"  [skip] developer_action={dev_action} -> not scorable, skipping LLM calls")
        not_scorable_judge = {
            "verdict": "NOT_SCORABLE",
            "reasoning": f"Ground truth action is {dev_action} — skipped before APA agent.",
            "confidence": 0.0,
            "evidence_used": [f"dev_action={dev_action}"],
        }
        return {
            "run_id": event.run_id,
            "repo": event.repo,
            "commit": event.commit_title[:60],
            "ground_truth": {
                "action": dev_action,
                "method": gt.classification_method,
                "reasoning": gt.developer_action_reasoning,
            },
            "rpa": {"prediction": rpa_result, "judge": not_scorable_judge},
            "apa_eig": {"prediction": {}, "judge": not_scorable_judge},
        }

    # 3. Run APA Agent (EIG Mode) — only if case is potentially scorable
    from src.apa.agent import (
        build_agent_graph,
        _initial_tools_for_event,
        _fetch_commit_diff,
        _summarize_patch_files,
        _fetch_run_history,
        _failed_step_summary,
    )
    from src.apa.bayesian_tracker import BeliefState

    # Prefetch the same zero-LLM evidence run_agent fetches before the graph:
    # commit diff (changed-file families) and run history (parent-run outcome
    # + recent failure rate, rendered into the deep_log_analysis header).
    changed_files = []
    commit_diff_summary = {}
    if event.repo and event.commit_sha:
        changed_files = _fetch_commit_diff(event.repo, event.commit_sha).get("files", [])
        commit_diff_summary = _summarize_patch_files(changed_files)
    run_history = {}
    if event.repo and event.branch and event.run_number:
        try:
            run_history = _fetch_run_history(
                repo=event.repo,
                branch=event.branch,
                workflow_path=event.workflow or "",
                current_run_number=event.run_number,
            )
        except Exception:
            pass

    # APA starts from the shared informed prior — NOT the RPA posterior.
    # Seeding with the RPA conclusion would bias every LLM step toward it.
    prior_bs = BeliefState()
    initial_state = {
        "run_event": kwargs,
        "raw_run": raw,
        "beliefs": dict(prior_bs.probabilities),
        "belief_history": list(prior_bs.history),
        "confidence": prior_bs.confidence(),
        "entropy": prior_bs.entropy(),
        "tools_available": _initial_tools_for_event(raw, event),
        "tools_called": [],
        "investigation_log": [],
        "current_step": 0,
        "done": False,
        "error_lines": error_lines,
        "mentioned_files": extraction.get("mentioned_files", []),
        "log_excerpt_texts": log_texts,
        "changed_files": changed_files,
        "commit_diff": commit_diff_summary,
        "failed_step_context": _failed_step_summary(event),
        "dependency_changes": {},
        "run_history": run_history,
        "similar_failures": [],
        "workflow_contents": [],
        "runner_environment": {},
        "pr_context": {},
        "semantic_diff_links": [],
        "preprocessing_summary": {"top_category": tracker.result()["category"]},
        "classification": {},
        "api_key": "",
        "model": MODEL_HARD,  # agent always uses the powerful model for real log analysis
        "_next_tool": "",
    }
    
    agent = build_agent_graph()
    apa_state = agent.invoke(initial_state)
    apa_result = apa_state.get("classification", {})

    # Post-run tier: use APA's confidence to decide judge model
    apa_confidence = apa_result.get("confidence", 0.0) if isinstance(apa_result, dict) else 0.0
    if apa_confidence >= CONFIDENCE_THRESHOLD:
        post_tier = "EASY"
        os.environ["CI_AGENT_MODEL"] = MODEL_EASY
        os.environ["JUDGE_MODEL"] = MODEL_EASY
    else:
        post_tier = "HARD"
        os.environ["CI_AGENT_MODEL"] = MODEL_HARD
        os.environ["JUDGE_MODEL"] = MODEL_HARD
    print(f"  [post-tier={post_tier}] apa_confidence={apa_confidence:.2f} -> judge model={os.environ['JUDGE_MODEL']}")

    # 4. Judge (only for scorable cases)
    rpa_judge = llm_judge(case_summary, rpa_result, gt, client)
    apa_judge = llm_judge(case_summary, apa_result, gt, client)

    return {
        "run_id": event.run_id,
        "repo": event.repo,
        "commit": event.commit_title[:60],
        "error_lines": error_lines[:8],
        "ground_truth": {
            "action": gt.developer_action,
            "method": gt.classification_method,
            "reasoning": gt.developer_action_reasoning
        },
        "rpa": {
            "prediction": rpa_result,
            "judge": rpa_judge
        },
        "apa_eig": {
            "prediction": apa_result,
            "judge": apa_judge
        }
    }


def main():
    client = make_client()
    
    print(f"Starting EIG vs RPA Benchmark for {N_CASES} cases...")
    print(f"Planner Mode: {os.environ['CI_AGENT_PLANNER_MODE']}")
    print(f"Model: {os.environ['CI_AGENT_MODEL']}")
    
    candidates = []
    seen_repos = set()
    
    # Load repos from OLD benchmark so we evaluate on fresh cases only
    old_repos: set[str] = set()
    if SEEN_REPOS_FROM_OLD_BENCHMARK.exists():
        try:
            old_data = json.loads(SEEN_REPOS_FROM_OLD_BENCHMARK.read_text())
            old_repos = {r.get("repo", "") for r in old_data}
            print(f"Excluding {len(old_repos)} repos from old benchmark for a fresh evaluation.")
        except Exception:
            pass

    # Load and Preflight
    print("\nScanning and Preflighting cases...")
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except: continue
            
            repo = run.get("intake", {}).get("repo", "") or run.get("repository_name", "")
            if not repo or repo in seen_repos or repo in old_repos: continue
            
            # Since the jsonl might already have preflight_action, check it
            preflight = run.get("preflight_action", "")
            if preflight and preflight not in SCORABLE_PREFLIGHT and preflight != "NO_PREFLIGHT":
                continue
                
            seen_repos.add(repo)
            candidates.append(run)
            if len(candidates) >= N_CASES:
                break

    print(f"Loaded {len(candidates)} candidate cases.")
    
    results = []
    processed_repos = set()
    qualitative_divergence = []

    # Resume from a previous partial run of THIS eval (if it crashed)
    if OUTPUT_PATH.exists():
        try:
            results = json.loads(OUTPUT_PATH.read_text())
            processed_repos = {r.get("repo", "") for r in results}
            print(f"Resuming fresh eval run: {len(results)} cases already done.")
        except Exception:
            pass

    if QUALITATIVE_PATH.exists():
        try:
            qualitative_divergence = json.loads(QUALITATIVE_PATH.read_text())
        except Exception:
            pass

    
    for i, raw in enumerate(candidates, 1):
        repo = raw.get("intake", {}).get("repo", "") or raw.get("repository_name", "")

        # Skip repos from old benchmark and already-processed repos in this run
        if repo in old_repos or repo in processed_repos:
            continue

            
        print(f"\n[{i}/{len(candidates)}] Processing {repo}...")
        
        try:
            res = run_case(raw, client)
            results.append(res)
            
            # Check for divergence
            rpa_v = res["rpa"]["judge"]["verdict"]
            apa_v = res["apa_eig"]["judge"]["verdict"]
            
            if (rpa_v == "CORRECT" and apa_v == "WRONG") or (apa_v == "CORRECT" and rpa_v == "WRONG"):
                qualitative_divergence.append(res)
                QUALITATIVE_PATH.write_text(json.dumps(qualitative_divergence, indent=2))
                
            # Incremental save
            OUTPUT_PATH.write_text(json.dumps(results, indent=2))
            
            # Stop early if we reach 300 scorable cases
            current_scorable = sum(
                1 for r in results 
                if r.get("rpa", {}).get("judge", {}).get("verdict") not in ("NOT_SCORABLE", "EVALUATION_ERROR", "")
                and r.get("apa_eig", {}).get("judge", {}).get("verdict") not in ("NOT_SCORABLE", "EVALUATION_ERROR", "")
            )
            if current_scorable >= 250:
                print(f"\nReached target of 250 scorable cases! Stopping.")
                break
            
        except Exception as e:
            print(f"  Error on {repo}: {e}")
            traceback.print_exc()

    # Calculate Metrics
    scorable = 0
    rpa_correct = 0
    apa_correct = 0
    apa_partial = 0
    not_scorable = 0
    eval_errors = 0

    for r in results:
        rpa_v = r["rpa"]["judge"]["verdict"]
        apa_v = r["apa_eig"]["judge"]["verdict"]

        # EVALUATION_ERROR = judge/scraper crashed — not the same as no GT
        if rpa_v == "EVALUATION_ERROR" or apa_v == "EVALUATION_ERROR":
            eval_errors += 1
            continue
            
        # NOT_SCORABLE = legitimate: no ground truth available
        if rpa_v == "NOT_SCORABLE" or apa_v == "NOT_SCORABLE":
            not_scorable += 1
            continue

        scorable += 1
        if rpa_v == "CORRECT": rpa_correct += 1
        if apa_v == "CORRECT": apa_correct += 1
        if apa_v == "PARTIAL": apa_partial += 1

    summary = {
        "total_cases": len(results),
        "scorable_cases": scorable,
        "not_scorable_cases": not_scorable,
        "evaluation_errors": eval_errors,
        "rpa_accuracy": round((rpa_correct / scorable * 100), 1) if scorable else 0,
        "apa_eig_accuracy": round((apa_correct / scorable * 100), 1) if scorable else 0,
        "apa_eig_accuracy_with_partial": round(((apa_correct + apa_partial) / scorable * 100), 1) if scorable else 0,
        "divergent_cases": len(qualitative_divergence)
    }
    
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    
    print("\n=== BENCHMARK COMPLETE ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
