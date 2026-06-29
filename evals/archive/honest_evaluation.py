# honest_evaluation.py
# ─────────────────────────────────────────────────────────────────────
# A clean, honest evaluation run on fresh data.
#
# No hand-tuned match functions. The LLM judges whether the
# classifier's recommendation aligns with developer behavior.
#
# Every judgment is saved with full reasoning so it's auditable.
# ─────────────────────────────────────────────────────────────────────

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

from src.apa.intake_parser import intake
from src.apa.log_extractor import extract_log_excerpt
from src.apa.classification_agent import classify
from ground_truth_scraper import scrape_ground_truth

load_dotenv()

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"
OUTPUT_PATH = Path("/home/guc_alaa/honest_eval_results.json")
N_CASES = 50
SCAN_RANGE = (100_000, 400_000)  # skip first 100k runs to get fresh repos
RANDOM_SEED = 99  # different seed than before

TOOLING_PATTERNS = (
    "bash-command-extractor", "Converting circular structure to JSON",
    "BashWord", "Parser exception",
)


def is_tooling_artifact(payload) -> bool:
    if not payload:
        return False
    text = json.dumps(payload) if not isinstance(payload, str) else payload
    return any(p in text for p in TOOLING_PATTERNS)


def is_good_candidate(run: dict) -> bool:
    meta = run.get("metadata") or {}
    if meta.get("conclusion") != "failure":
        return False
    if not (run.get("logs_archive") or {}).get("path"):
        return False
    insights = run.get("log_insights") or []
    if not insights:
        return False
    if not any(len(j.get("steps") or []) >= 2 for j in insights):
        return False
    has_real = False
    has_any = False
    for job in insights:
        for step in job.get("steps") or []:
            for key in ("error", "errors"):
                payload = step.get(key)
                if payload:
                    has_any = True
                    if not is_tooling_artifact(payload):
                        has_real = True
    if has_any and not has_real:
        return False
    return True


def tarball_path(raw: dict) -> str:
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/"):] if p.startswith("/data/") else p


# ─── LLM judge ───────────────────────────────────────────────────────

def llm_judge(
    case_summary: str,
    classifier_output: dict,
    ground_truth: dict,
    client: OpenAI,
) -> dict:
    """
    Use DeepSeek to judge whether the classifier's recommendation
    aligns with what the developer actually did.
    
    Returns {"verdict": "MATCH|PARTIAL|MISMATCH|NOT_SCORABLE",
             "reasoning": "..."}
    """
    prompt = f"""You are evaluating an AI system that classifies CI/CD failures and recommends actions.

THE CI FAILURE:
{case_summary}

THE AI SYSTEM RECOMMENDED:
  Category: {classifier_output.get('category')}
  Severity: {classifier_output.get('severity')}
  Action:   {classifier_output.get('action')}
  Reasoning: {classifier_output.get('reasoning')}

WHAT THE DEVELOPER ACTUALLY DID:
  Action: {ground_truth.get('developer_action')}
  Method of determination: {ground_truth.get('method')}
  Details: {ground_truth.get('reasoning')}

IMPORTANT CONTEXT FOR JUDGING:
- "NO_IMMEDIATE_RESPONSE" means no code changes within 7 days. This could mean the developer retried (no commit needed), ignored it, or the failure was on a low-priority branch. It does NOT necessarily mean the AI was wrong.
- "PR_MERGED" means the developer merged despite the failure. This could mean they judged the failure as non-blocking, or they fixed it in a separate commit.
- "PR_CLOSED_UNMERGED" means they rejected the change entirely.
- "UNKNOWN" means we could not determine what happened.

YOUR TASK:
Judge whether the AI's recommendation was reasonable given what the developer actually did. Choose one:

MATCH — The AI's recommendation is well-aligned with the developer's action. The AI would have led to a good outcome.
PARTIAL — The AI's recommendation is somewhat aligned but missed important nuance, or the developer's action is ambiguous enough that we can't be sure.
MISMATCH — The AI's recommendation clearly conflicts with what the developer did, and the developer's action suggests the AI was wrong.
NOT_SCORABLE — The developer's action is UNKNOWN or there isn't enough information to judge.

Respond with ONLY a JSON object:
{{
  "verdict": "MATCH or PARTIAL or MISMATCH or NOT_SCORABLE",
  "reasoning": "2-3 sentences explaining your judgment, referencing specific aspects of both the AI output and the developer action"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an impartial evaluator of AI systems. Be fair but honest. Don't inflate scores."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"verdict": "NOT_SCORABLE", "reasoning": f"Judge failed: {e}"}


# ─── pipeline runner ─────────────────────────────────────────────────

def run_full_pipeline(raw: dict, client: OpenAI) -> dict:
    event = intake(raw)
    tarball_name = tarball_path(raw)

    # Extract logs
    excerpts = []
    strategies = []
    sample_error_lines = []
    total_lines = 0
    error_markers_total = 0

    for fs in event.failed_steps:
        ex = extract_log_excerpt(
            zip_path=ZIP_PATH,
            tarball_name=tarball_name,
            job_file=fs.job_file,
            step_label=fs.step_label or "",
        )
        excerpts.append(ex)
        strategies.append(ex.strategy_used)
        total_lines += sum(len(w) for w in ex.error_windows)
        error_markers_total += len(ex.error_marker_lines)
        for ml in ex.error_marker_lines[:2]:
            cleaned = ml.strip()[:120]
            if cleaned not in sample_error_lines:
                sample_error_lines.append(cleaned)
        for w in ex.error_windows:
            for ln in w[-10:]:
                ln_clean = ln.strip()
                if any(kw in ln_clean.lower() for kw in
                       ("error", "failed", "fatal", "exception", "not found")):
                    if ln_clean not in sample_error_lines and len(sample_error_lines) < 5:
                        sample_error_lines.append(ln_clean[:120])

    # Classify
    cl_result = classify(event, client, log_excerpts=excerpts if excerpts else None)

    # Ground truth
    gt = scrape_ground_truth(event)

    # LLM judge
    case_summary = (
        f"Repo: {event.repo}\n"
        f"Workflow: {event.workflow}\n"
        f"Branch: {event.branch} (protected: {event.is_protected_branch})\n"
        f"Commit: {event.commit_title}\n"
        f"Event: {event.event}\n"
        f"Failed jobs: {event.failed_jobs_count}/{event.n_jobs}\n"
        f"Detection: {event.failure_detection}\n"
        f"Key errors from logs: {'; '.join(sample_error_lines[:3])}"
    )

    judge = llm_judge(
        case_summary,
        asdict(cl_result),
        {
            "developer_action": gt.developer_action,
            "method": gt.classification_method,
            "reasoning": gt.developer_action_reasoning,
        },
        client,
    )

    return {
        "case_label": f"{event.repo} — {event.commit_title[:60]}",
        "intake": {
            "run_id": event.run_id,
            "repo": event.repo,
            "workflow": event.workflow,
            "branch": event.branch,
            "event": event.event,
            "is_protected_branch": event.is_protected_branch,
            "commit_sha": event.commit_sha,
            "commit_title": event.commit_title,
            "conclusion": event.conclusion,
            "failure_detection": event.failure_detection,
            "failed_jobs_count": event.failed_jobs_count,
            "n_jobs": event.n_jobs,
            "all_failures_are_tooling_artifacts": event.all_failures_are_tooling_artifacts,
        },
        "extraction": {
            "total_steps_extracted": len(excerpts),
            "strategies": strategies,
            "total_lines_extracted": total_lines,
            "error_markers_found": error_markers_total,
            "sample_error_lines": sample_error_lines[:5],
        },
        "classification": {
            "category": cl_result.category,
            "severity": cl_result.severity,
            "confidence": cl_result.confidence,
            "action": cl_result.action,
            "reasoning": cl_result.reasoning,
            "evidence": cl_result.evidence,
            "unknowns": cl_result.unknowns,
        },
        "ground_truth": {
            "developer_action": gt.developer_action,
            "method": gt.classification_method,
            "reasoning": gt.developer_action_reasoning,
            "match_verdict": judge.get("verdict", "NOT_SCORABLE"),
            "judge_reasoning": judge.get("reasoning", ""),
        },
    }


# ─── main ────────────────────────────────────────────────────────────

def main():
    from llm_config import make_client

    try:
        client = make_client()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    # Collect candidates from a DIFFERENT part of the dataset
    print(f"Scanning runs {SCAN_RANGE[0]:,}–{SCAN_RANGE[1]:,} for candidates...")
    candidates = []
    seen_repos = set()
    scanned = 0

    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned < SCAN_RANGE[0]:
                continue
            if scanned > SCAN_RANGE[1]:
                break
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not is_good_candidate(run):
                continue

            repo = run.get("repository_name", "")
            if repo in seen_repos:
                continue
            seen_repos.add(repo)
            candidates.append(run)

            if scanned % 50_000 == 0:
                print(f"  at run {scanned:,}, {len(candidates)} candidates")

            if len(candidates) >= N_CASES * 3:
                break

    print(f"  {len(candidates)} candidates from {len(seen_repos)} unique repos")

    random.seed(RANDOM_SEED)
    selected = random.sample(candidates, min(N_CASES, len(candidates)))
    print(f"  selected {len(selected)} for evaluation\n")

    # Run pipeline
    results = []
    for i, raw in enumerate(selected, 1):
        repo = raw.get("repository_name", "?")
        commit = ((raw.get("metadata") or {}).get("head_commit") or {}).get("message", "").split("\n")[0]
        print(f"[{i}/{len(selected)}] {repo} — {commit[:50]}")

        try:
            result = run_full_pipeline(raw, client)
            results.append(result)
            v = result["ground_truth"]["match_verdict"]
            cat = result["classification"]["category"]
            print(f"  → {cat} | judge: {v}")
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")
            traceback.print_exc()

        # Save incrementally
        OUTPUT_PATH.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE: {len(results)} cases")
    print(f"Results: {OUTPUT_PATH}\n")

    verdicts = [r["ground_truth"]["match_verdict"] for r in results]
    for v in ["MATCH", "PARTIAL", "MISMATCH", "NOT_SCORABLE"]:
        count = verdicts.count(v)
        pct = count * 100 // len(verdicts) if verdicts else 0
        print(f"  {v:<14} {count:>3}  ({pct}%)")

    scorable = [v for v in verdicts if v not in ("NOT_SCORABLE",)]
    if scorable:
        good = sum(1 for v in scorable if v in ("MATCH", "PARTIAL"))
        print(f"\n  Scorable:        {len(scorable)}")
        print(f"  Match+Partial:   {good} ({good*100//len(scorable)}%)")
        print(f"  Mismatch:        {len(scorable)-good} ({(len(scorable)-good)*100//len(scorable)}%)")

    cats = [r["classification"]["category"] for r in results]
    print(f"\nCategories:")
    for cat in sorted(set(cats)):
        print(f"  {cat}: {cats.count(cat)}")

    # Ground-truth attribution source summary
    methods = [r["ground_truth"].get("method", "none") for r in results]
    llm_methods = {"llm"}
    script_methods = {"rubric", "pr_state", "rerun", "none"}

    llm_total = sum(1 for m in methods if m in llm_methods)
    script_total = sum(1 for m in methods if m in script_methods)
    other_total = len(results) - llm_total - script_total

    print(f"\nGround-truth source:")
    print(f"  LLM:    {llm_total}")
    print(f"  Script: {script_total}")
    if other_total:
        print(f"  Other:  {other_total}")

    # "Matched" split by source (MATCH only, and MATCH+PARTIAL as aligned)
    match_llm = sum(
        1
        for r in results
        if r["ground_truth"].get("match_verdict") == "MATCH"
        and r["ground_truth"].get("method") in llm_methods
    )
    match_script = sum(
        1
        for r in results
        if r["ground_truth"].get("match_verdict") == "MATCH"
        and r["ground_truth"].get("method") in script_methods
    )
    aligned_llm = sum(
        1
        for r in results
        if r["ground_truth"].get("match_verdict") in ("MATCH", "PARTIAL")
        and r["ground_truth"].get("method") in llm_methods
    )
    aligned_script = sum(
        1
        for r in results
        if r["ground_truth"].get("match_verdict") in ("MATCH", "PARTIAL")
        and r["ground_truth"].get("method") in script_methods
    )

    print(f"\nMatched by source (judge verdict):")
    print(f"  MATCH only      -> LLM: {match_llm}, Script: {match_script}")
    print(f"  MATCH+PARTIAL   -> LLM: {aligned_llm}, Script: {aligned_script}")

    method_counts = Counter(methods)
    print(f"\nGround-truth methods (detailed):")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")


if __name__ == "__main__":
    main()