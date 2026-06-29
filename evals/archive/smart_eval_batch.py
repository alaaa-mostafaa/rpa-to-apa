"""
smart_eval_batch.py

Finds 40 high-quality eval cases by:
  1. Filtering for protected-branch failures (main/master/develop) with real logs
  2. Deduplicating by repo (one case per repo)
  3. Running classification + ground truth scraping together
  4. Running the diagnosis-quality judge

This avoids the UNKNOWN ground truth problem by focusing on branches
that are never deleted, so follow-up commits are always findable.
"""

import gzip
import json
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, "/home/guc_alaa")

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"
OUTPUT_ROOT = Path("/home/guc_alaa/comparison_results")
N_TARGET = 100
SCAN_OFFSET = 250_000  # skip first N lines to get fresh repos not seen in previous runs
SCAN_LIMIT = 500_000

PROTECTED_BRANCHES = {"main", "master", "develop", "dev", "release"}

TOOLING_PATTERNS = (
    "bash-command-extractor", "Converting circular structure to JSON",
    "BashWord", "Parser exception",
)


def is_tooling_artifact(payload) -> bool:
    if not payload:
        return False
    text = json.dumps(payload) if not isinstance(payload, str) else payload
    return any(p in text for p in TOOLING_PATTERNS)


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


def repo_from_run_id(run_id: str) -> str:
    m = re.match(r"^(.+?)_\.github/", run_id)
    return m.group(1) if m else ""


# ── judge ────────────────────────────────────────────────────────────

def llm_judge(case_summary: str, classifier_output: dict, ground_truth, client) -> dict:
    from dataclasses import asdict, is_dataclass
    if is_dataclass(ground_truth):
        gt = asdict(ground_truth)
    else:
        gt = ground_truth if isinstance(ground_truth, dict) else {}

    dev_action = gt.get("developer_action", "UNKNOWN")

    unscoreable = {"UNKNOWN", "PR_MERGED", "PR_MERGED_UNCLEAR", "PR_STILL_OPEN", "UNRELATED", "EVALUATION_ERROR", ""}
    if dev_action in unscoreable:
        return {
            "verdict": "NOT_SCORABLE",
            "reasoning": f"Ground truth is '{dev_action}' — cannot determine what was actually wrong.",
        }

    ai_diag = (
        f"Category: {classifier_output.get('category')}\n"
        f"Reasoning: {str(classifier_output.get('reasoning', ''))[:500]}\n"
        f"Evidence: {classifier_output.get('evidence', [])}"
    )
    fix = classifier_output.get("fix_suggestion")
    if fix:
        ai_diag += f"\nFix suggestion: file={fix.get('file')} change={str(fix.get('change',''))[:150]}"

    # Build a readable ground truth summary
    gt_details = gt.get("developer_action_reasoning", gt.get("reasoning", ""))
    # Include follow-up commit file names if available
    follow_ups = gt.get("follow_up_commits", [])
    if follow_ups and isinstance(follow_ups, list) and len(follow_ups) > 0:
        for c in follow_ups[:2]:
            files = getattr(c, "files_changed", None) or c.get("files_changed", [])
            if files:
                gt_details += f" | files: {', '.join(files[:8])}"

    prompt = f"""You are evaluating whether an AI correctly diagnosed a CI/CD failure.

THE CI FAILURE:
{case_summary}

AI DIAGNOSIS:
{ai_diag}

WHAT THE DEVELOPER ACTUALLY FIXED:
  Type of change: {dev_action}
  Details: {str(gt_details)[:500]}

YOUR TASK: Judge whether the AI's diagnosis of the root cause matches what the developer actually fixed.
Focus on the PROBLEM the AI identified vs the PROBLEM the developer fixed — not the label names.

SEMANTIC EQUIVALENCES — treat these as CORRECT matches:
  AI diagnoses CONFIG_ERROR (workflow/CI config/permissions/runner issue) → dev_action = WORKFLOW_FIX
  AI diagnoses DEPENDENCY_CONFLICT (version mismatch/unresolved dep) → dev_action = PIN_VERSION or DEPENDENCY_CHANGE
  AI diagnoses CODE_REGRESSION (source code bug) → dev_action = CODE_FIX or CODE_CHANGE
  AI diagnoses INFRA_INCOMPATIBILITY (runner/OS/tool incompatibility) → dev_action = WORKFLOW_FIX or PIN_VERSION

PARTIAL matches — right problem space, wrong specific cause:
  AI diagnoses CONFIG_ERROR but dev_action = PIN_VERSION: PARTIAL (both are non-code infrastructure fixes; AI saw a workflow-level symptom but missed that the root cause was a version)
  AI diagnoses DEPENDENCY_CONFLICT but dev_action = WORKFLOW_FIX: PARTIAL (overlapping infrastructure space)

CORRECT — AI identified the actual root cause (apply semantic equivalences above).
  Examples: AI says "numpy version conflict" → dev pinned numpy; AI says "workflow uses deprecated action" → dev updated that action; AI says "CONFIG_ERROR: permissions in workflow" → dev edited workflow YAML.

WRONG — AI misidentified the cause. Its reasoning points to a clearly different problem than what was fixed.
  Examples: AI blames source code logic → dev fixed a workflow YAML; AI says "flaky infra/transient" → dev had to change actual code or deps.

PARTIAL — AI identified the right category/area but the specific diagnosis is off (right file family, wrong issue within it).
  Examples: AI says "CONFIG_ERROR: missing secret" but dev fixed a goreleaser version incompatibility in the workflow.

NOT_SCORABLE — The ground truth doesn't reveal enough about what specifically was broken, or dev_action is NO_IMMEDIATE_RESPONSE.

Respond with ONLY a JSON object:
{{
  "verdict": "CORRECT or WRONG or PARTIAL or NOT_SCORABLE",
  "reasoning": "1-2 sentences: what the AI said vs what the developer actually fixed."
}}"""

    try:
        from llm_config import get_provider
        provider = get_provider()
        if provider == "deepseek":
            model = "deepseek-chat"
        elif provider == "openrouter":
            model = "openai/gpt-4o-mini" # OpenRouter uses a prefix for OpenAI models
        else:
            model = "gpt-4o-mini"
            
        from llm_usage import record_usage
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict evaluator of AI diagnostic quality. Do not give credit for vague or generic reasoning."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
        )
        record_usage(response, model, call_type="chat", label="smart_eval.judge")
        data = json.loads(response.choices[0].message.content)
        return {"verdict": data.get("verdict", "EVALUATION_ERROR"), "reasoning": data.get("reasoning", "")}
    except Exception as e:
        return {"verdict": "EVALUATION_ERROR", "reasoning": f"{type(e).__name__}: {e}"}


# ── main ─────────────────────────────────────────────────────────────

SCORABLE_PREFLIGHT = {"CODE_FIX", "WORKFLOW_FIX", "PIN_VERSION", "REVERT", "CODE_CHANGE", "PR_MERGED"}


def main():
    from agent import run_agent
    from intake_parser import intake
    from log_extractor import extract_log_excerpt
    from ground_truth_scraper import scrape_ground_truth, classify_pr_fix_type, fetch_pr_files, quick_gt_preflight
    from llm_config import make_client

    client = make_client()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"smart_eval_{N_TARGET}_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    print(f"Scanning up to {SCAN_LIMIT:,} runs, looking for {N_TARGET} cases with scorable ground truth...")

    candidates = []
    seen_repos = set()
    scanned = 0
    preflight_checked = 0
    preflight_passed = 0

    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned <= SCAN_OFFSET:
                continue
            if scanned > SCAN_LIMIT:
                break
            try:
                run = json.loads(line)
            except:
                continue
            if not is_good_candidate(run):
                continue
            repo = run.get("repository_name", "")
            if repo in seen_repos:
                continue

            # Pre-flight: cheap GitHub API check — skip NO_IMMEDIATE_RESPONSE cases
            meta = run.get("metadata") or {}
            sha = (meta.get("head_commit") or {}).get("id", "")
            branch = meta.get("head_branch", "")
            started_at = meta.get("created_at", "") or meta.get("run_started_at", "")
            parts = repo.split("/")
            if sha and len(parts) == 2:
                preflight_checked += 1
                action = quick_gt_preflight(parts[0], parts[1], sha, branch, started_at)
                if action not in SCORABLE_PREFLIGHT:
                    continue  # skip transient/no-response cases
                preflight_passed += 1
                if preflight_passed % 5 == 0:
                    print(f"  preflight: {preflight_passed} passed / {preflight_checked} checked (line {scanned:,})")

            seen_repos.add(repo)
            candidates.append(run)
            if len(candidates) >= N_TARGET:
                break

    print(f"\nPreflight: {preflight_passed}/{preflight_checked} passed ({scanned:,} lines scanned)")
    print(f"Candidates: {len(candidates)} from {len(seen_repos)} unique repos\n")

    if not candidates:
        print("No candidates found.")
        return

    results = []
    verdict_counts = Counter()

    for i, raw in enumerate(candidates, 1):
        run_id = raw.get("_id", "")
        repo = raw.get("repository_name", "")
        meta = raw.get("metadata") or {}
        branch = meta.get("head_branch", "")
        commit = (meta.get("head_commit") or {}).get("message", "").split("\n")[0][:60]

        print(f"\n[{i:>2}/{len(candidates)}] {repo}  branch={branch}")
        print(f"         commit: {commit}")

        result_entry = {
            "index": i,
            "run_id": run_id,
            "repo": repo,
            "branch": branch,
            "commit": commit,
            "classification": {},
            "ground_truth": {},
            "judge": {"verdict": "EVALUATION_ERROR", "reasoning": ""},
        }

        # ── Step 1: classify ─────────────────────────────────────────
        try:
            agent_result = run_agent(raw)
            cl = agent_result.get("classification", {})
            result_entry["classification"] = cl

            # Check for 402 error
            if "402" in str(cl.get("reasoning", "")):
                print(f"  ✗ classifier hit 402 credit error — skipping")
                results.append(result_entry)
                output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                continue

            print(f"  classifier: {cl.get('category')} | conf={cl.get('confidence', 0):.2f}")
        except Exception as e:
            print(f"  ✗ classify error: {type(e).__name__}: {e}")
            results.append(result_entry)
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            continue

        # ── Step 2: ground truth ─────────────────────────────────────
        try:
            event = intake(raw)
            gt = scrape_ground_truth(event)

            # Upgrade PR_MERGED with actual file list
            if gt.developer_action == "PR_MERGED":
                pr_num_match = re.search(r"PR #(\d+)", gt.developer_action_reasoning)
                parts = repo.split("/")
                if pr_num_match and len(parts) == 2:
                    pr_num = int(pr_num_match.group(1))
                    pr_files = fetch_pr_files(parts[0], parts[1], pr_num)
                    if pr_files:
                        action, reasoning, signals = classify_pr_fix_type(
                            {"title": gt.developer_action_reasoning, "body": ""}, pr_files
                        )
                        gt.developer_action = action
                        gt.developer_action_reasoning = f"[pr_files] {reasoning}"
                        gt.classification_method = "pr_files"

            from dataclasses import asdict
            result_entry["ground_truth"] = asdict(gt)
            print(f"  ground truth: {gt.developer_action} (via {gt.classification_method})")
        except Exception as e:
            print(f"  ✗ ground truth error: {type(e).__name__}: {e}")
            results.append(result_entry)
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            continue

        # ── Step 3: judge ────────────────────────────────────────────
        case_summary = (
            f"Repo: {repo}  Branch: {branch} (protected)\n"
            f"Commit: {commit}\n"
            f"Workflow: {event.workflow}\n"
            f"Failed jobs: {event.failed_jobs_count}/{event.n_jobs}\n"
            f"Failure detection: {event.failure_detection}"
        )

        judge = llm_judge(case_summary, cl, gt, client)
        result_entry["judge"] = judge
        verdict_counts[judge["verdict"]] += 1
        print(f"  judge: {judge['verdict']} — {judge['reasoning'][:100]}")

        results.append(result_entry)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ── summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"DONE: {len(results)} cases from {len(set(r['repo'] for r in results))} unique repos")
    print(f"Output: {output_path}\n")

    total = len(results)
    for v in ["CORRECT", "PARTIAL", "WRONG", "NOT_SCORABLE", "EVALUATION_ERROR"]:
        n = verdict_counts[v]
        print(f"  {v:<18} {n:>3}  ({n*100//total if total else 0}%)")

    scorable = [r for r in results if r["judge"]["verdict"] in ("CORRECT", "PARTIAL", "WRONG")]
    if scorable:
        n_s = len(scorable)
        correct = sum(1 for r in scorable if r["judge"]["verdict"] == "CORRECT")
        partial = sum(1 for r in scorable if r["judge"]["verdict"] == "PARTIAL")
        wrong = sum(1 for r in scorable if r["judge"]["verdict"] == "WRONG")
        print(f"\n  Scorable: {n_s}")
        print(f"  CORRECT:  {correct} ({correct*100//n_s}%)")
        print(f"  PARTIAL:  {partial} ({partial*100//n_s}%)")
        print(f"  WRONG:    {wrong} ({wrong*100//n_s}%)")

    gt_dist = Counter(r["ground_truth"].get("developer_action", "?") for r in results)
    print(f"\nGround truth distribution:")
    for k, v in gt_dist.most_common():
        print(f"  {k}: {v}")

    credit_errors = sum(1 for r in results if "402" in str(r["classification"].get("reasoning", "")))
    if credit_errors:
        print(f"\n  ⚠ {credit_errors} cases skipped due to 402 credit errors")


if __name__ == "__main__":
    main()
