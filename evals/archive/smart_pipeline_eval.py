import gzip
import json
import os
import random
import traceback
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI
import sys

# Ensure root directory is in path
sys.path.insert(0, "/home/guc_alaa")
try:
    from src.apa.intake_parser import intake
    from src.apa.log_extractor import extract_log_excerpt
    from src.apa.classification_agent import classify
    from ground_truth_scraper import scrape_ground_truth
    from src.apa.bayesian_tracker_dual import DualTracker
except ImportError:
    # Fallback if running locally
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.apa.intake_parser import intake
    from src.apa.log_extractor import extract_log_excerpt
    from src.apa.classification_agent import classify
    from ground_truth_scraper import scrape_ground_truth
    from src.apa.bayesian_tracker_dual import DualTracker

load_dotenv()

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"
OUTPUT_PATH = Path("/home/guc_alaa/smart_eval_results.json")
SUMMARY_PATH = Path("/home/guc_alaa/smart_eval_summary.json")

N_CASES = 100
SCAN_RANGE = (100_000, 500_000)
RANDOM_SEED = 42

PROTECTED_BRANCHES = {"main", "master", "develop", "dev", "release"}

def is_protected(branch: str) -> bool:
    b = (branch or "").lower()
    return b in PROTECTED_BRANCHES or b.startswith("release/") or b.startswith("main-") or b.startswith("v")

def is_good_candidate(run: dict) -> bool:
    meta = run.get("metadata") or {}
    if meta.get("conclusion") != "failure":
        return False
    branch = meta.get("head_branch", "")
    if not is_protected(branch):
        return False
    if not (run.get("logs_archive") or {}).get("path"):
        return False
    insights = run.get("log_insights") or []
    if not insights:
        return False
    if not any(len(j.get("steps") or []) >= 2 for j in insights):
        return False
    return True

def tarball_path(raw: dict) -> str:
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/"):] if p.startswith("/data/") else p

def get_error_lines(excerpts: list) -> list:
    error_lines = []
    for ex in excerpts:
        for ml in ex.error_marker_lines[:2]:
            cleaned = ml.strip()[:120]
            if cleaned and cleaned not in error_lines:
                error_lines.append(cleaned)
        for w in ex.error_windows:
            for ln in w[-5:]:
                ln_clean = ln.strip()
                if any(kw in ln_clean.lower() for kw in ("error", "failed", "fatal", "exception")):
                    if ln_clean not in error_lines and len(error_lines) < 5:
                        error_lines.append(ln_clean[:120])
    return error_lines

def smart_llm_judge(case_summary: str, ai_output: dict, ground_truth: dict, client: OpenAI) -> dict:
    prompt = f"""You are an expert software engineer evaluating an AI system that triages CI/CD failures.

THE CI FAILURE:
{case_summary}

THE AI SYSTEM PREDICTED:
  Category: {ai_output.get('category')}
  Reasoning: {ai_output.get('reasoning') or 'None provided (Bayesian mode)'}

WHAT THE DEVELOPER ACTUALLY DID:
  Action: {ground_truth.get('developer_action')}
  Method: {ground_truth.get('method')}
  Details: {ground_truth.get('reasoning')}

IMPORTANT CONTEXT:
- "NO_IMMEDIATE_RESPONSE": No code changes within 7 days. This could mean they retried it, ignored it, or it was low priority.
- "PR_MERGED": They merged despite the failure.
- "PR_CLOSED_UNMERGED": They rejected the change.
- "UNRELATED": Follow-up commits had nothing to do with the failure.
- "UNKNOWN": Cannot be determined.

YOUR TASK:
Judge the AI's prediction accuracy. Do NOT use a vague "PARTIAL". Be decisive.
Choose ONE of the following verdicts:

EXACT_MATCH: The AI's predicted category and reasoning perfectly align with what the developer did.
PLAUSIBLE_ALTERNATIVE: The developer did something else (or did nothing/ignored it), but the AI's diagnosis is a highly plausible, correct engineering assessment of the failure based on the logs.
MISMATCH: The AI's diagnosis is incorrect based on the logs and developer action.
NOT_SCORABLE: The developer action is UNKNOWN or UNRELATED, AND the log evidence is too weak to determine if the AI was right.

Respond with ONLY a JSON object:
{{
  "verdict": "EXACT_MATCH" or "PLAUSIBLE_ALTERNATIVE" or "MISMATCH" or "NOT_SCORABLE",
  "reasoning": "Short explanation of your judgment."
}}"""

    try:
        from src.apa.llm_config import get_provider
        provider = get_provider()
        if provider == "deepseek":
            model_name = "deepseek-chat"
        elif provider == "openrouter":
            model_name = "openai/gpt-4o-mini"
        else:
            model_name = "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict and highly capable technical judge."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"verdict": "NOT_SCORABLE", "reasoning": f"Judge failed: {e}"}

def run_case(raw: dict, client: OpenAI) -> dict:
    event = intake(raw)
    tarball_name = tarball_path(raw)

    excerpts = []
    for fs in event.failed_steps[:3]:
        ex = extract_log_excerpt(ZIP_PATH, tarball_name, fs.job_file, fs.step_label or "")
        excerpts.append(ex)

    error_lines = get_error_lines(excerpts)

    # RPA Tracker
    tracker = DualTracker(mode="rpa", client=None)
    tracker.observe_branch(event.is_protected_branch, event.branch)
    tracker.observe_jobs(event.failed_jobs_count, event.n_jobs)
    if error_lines: tracker.observe_errors(error_lines)
    tracker.observe_commit(event.commit_message or event.commit_title)
    tracker.observe_detection(event.failure_detection)
    rpa_result = tracker.result()

    from src.apa.llm_config import get_provider
    provider = get_provider()
    if provider == "deepseek":
        model_name = "deepseek-chat"
    elif provider == "openrouter":
        model_name = "openai/gpt-4o-mini"
    else:
        model_name = "gpt-4o-mini"

    # APA Classification
    apa_result = classify(event, client, model=model_name, log_excerpts=excerpts if excerpts else None)

    # Ground Truth
    gt = scrape_ground_truth(event)
    gt_dict = {
        "developer_action": gt.developer_action,
        "method": gt.classification_method,
        "reasoning": gt.developer_action_reasoning,
    }

    case_summary = (
        f"Repo: {event.repo}\n"
        f"Workflow: {event.workflow}\n"
        f"Branch: {event.branch}\n"
        f"Commit: {event.commit_title}\n"
        f"Key errors: {'; '.join(error_lines[:3])}"
    )

    rpa_judge = smart_llm_judge(case_summary, rpa_result, gt_dict, client)
    apa_judge = smart_llm_judge(case_summary, asdict(apa_result), gt_dict, client)

    return {
        "run_id": event.run_id,
        "repo": event.repo,
        "commit": event.commit_title[:60],
        "ground_truth": gt_dict,
        "rpa": {
            "prediction": rpa_result,
            "judge": rpa_judge
        },
        "apa": {
            "prediction": asdict(apa_result),
            "judge": apa_judge
        }
    }

def main():
    from src.apa.llm_config import make_client
    client = make_client()

    print("Scanning for candidates...")
    candidates = []
    seen_repos = set()
    scanned = 0
    preflight_checked = 0
    preflight_passed = 0
    
    from ground_truth_scraper import quick_gt_preflight
    SCORABLE_PREFLIGHT = {"CODE_FIX", "WORKFLOW_FIX", "PIN_VERSION", "REVERT", "CODE_CHANGE", "PR_MERGED"}
    
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned < SCAN_RANGE[0]: continue
            if scanned > SCAN_RANGE[1]: break
            try:
                run = json.loads(line)
            except: continue
            
            repo = run.get("repository_name", "")
            if repo in seen_repos: continue
            
            if is_good_candidate(run):
                # Cheap GitHub API pre-flight to skip NO_IMMEDIATE_RESPONSE cases early
                meta = run.get("metadata") or {}
                sha = (meta.get("head_commit") or {}).get("id", "")
                branch = meta.get("head_branch", "")
                started_at = meta.get("created_at", "") or meta.get("run_started_at", "")
                parts = repo.split("/")
                if sha and len(parts) == 2:
                    preflight_checked += 1
                    action = quick_gt_preflight(parts[0], parts[1], sha, branch, started_at)
                    if action not in SCORABLE_PREFLIGHT:
                        continue  # Skip transient or ignored failures
                    
                    preflight_passed += 1
                    if preflight_passed % 5 == 0:
                        print(f"  preflight: {preflight_passed} valid repos found / {preflight_checked} API checks (line {scanned:,})")
                        
                candidates.append(run)
                seen_repos.add(repo)
            if len(candidates) >= N_CASES: break

    print(f"\nScan complete: {preflight_passed} valid candidates found from {preflight_checked} repos checked.")

    random.seed(RANDOM_SEED)
    selected = random.sample(candidates, min(N_CASES, len(candidates)))
    print(f"Running evaluation on {len(selected)} cases...")

    results = []
    for i, raw in enumerate(selected, 1):
        try:
            print(f"[{i}/{len(selected)}] {raw.get('repository_name')}")
            res = run_case(raw, client)
            results.append(res)
            
            # Save incrementally
            OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"  Error: {e}")

    # Metrics computation
    total = len(results)
    rpa_correct = 0
    apa_correct = 0
    scorable = 0

    apa_wins = []
    rpa_wins = []

    for r in results:
        rpa_v = r["rpa"]["judge"]["verdict"]
        apa_v = r["apa"]["judge"]["verdict"]
        
        if rpa_v != "NOT_SCORABLE" or apa_v != "NOT_SCORABLE":
            scorable += 1

        rpa_ok = rpa_v in ("EXACT_MATCH", "PLAUSIBLE_ALTERNATIVE")
        apa_ok = apa_v in ("EXACT_MATCH", "PLAUSIBLE_ALTERNATIVE")

        if rpa_ok: rpa_correct += 1
        if apa_ok: apa_correct += 1

        if apa_ok and not rpa_ok:
            apa_wins.append({
                "repo": r["repo"],
                "apa_cat": r["apa"]["prediction"]["category"],
                "rpa_cat": r["rpa"]["prediction"]["category"],
                "apa_reasoning": r["apa"]["judge"]["reasoning"]
            })
        elif rpa_ok and not apa_ok:
            rpa_wins.append({
                "repo": r["repo"],
                "apa_cat": r["apa"]["prediction"]["category"],
                "rpa_cat": r["rpa"]["prediction"]["category"],
                "rpa_reasoning": r["rpa"]["judge"]["reasoning"]
            })

    summary = {
        "total_cases": total,
        "scorable_cases": scorable,
        "rpa_accuracy": round(rpa_correct / scorable * 100, 1) if scorable else 0,
        "apa_accuracy": round(apa_correct / scorable * 100, 1) if scorable else 0,
        "apa_wins_count": len(apa_wins),
        "rpa_wins_count": len(rpa_wins),
        "apa_wins_examples": apa_wins[:10],
        "rpa_wins_examples": rpa_wins[:10]
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    
    print("\n=== EVALUATION COMPLETE ===")
    print(f"Scorable cases: {scorable}/{total}")
    print(f"RPA Accuracy: {summary['rpa_accuracy']}%")
    print(f"APA Accuracy: {summary['apa_accuracy']}%")
    print(f"APA won over RPA in {len(apa_wins)} cases.")
    print(f"RPA won over APA in {len(rpa_wins)} cases.")

if __name__ == "__main__":
    main()
