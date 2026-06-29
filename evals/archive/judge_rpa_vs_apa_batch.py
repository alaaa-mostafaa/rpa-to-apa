import argparse
import gzip
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, "/home/guc_alaa")

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")

from ground_truth_scraper import scrape_ground_truth, classify_pr_fix_type, fetch_pr_files
from src.apa.intake_parser import intake


TRUTH_ACTION_TO_CATEGORY = {
    "CODE_FIX": "CODE_REGRESSION",
    "CODE_CHANGE": "CODE_REGRESSION",
    "REVERT": "CODE_REGRESSION",
    "WORKFLOW_FIX": "CONFIG_ERROR",
    "PIN_VERSION": "DEPENDENCY_CONFLICT",
    "DEPENDENCY_CHANGE": "DEPENDENCY_CONFLICT",
    "RETRY": "ENV_FLAKINESS",
    "RETRY_SUCCEEDED": "ENV_FLAKINESS",
}


UNSCOREABLE_TRUTH_ACTIONS = {
    "UNRELATED",
    "UNKNOWN",
    "NO_IMMEDIATE_RESPONSE",
    "PR_STILL_OPEN",
    "PR_CLOSED_UNMERGED",
    "PR_MERGED",
    "EVALUATION_ERROR",
    "",
}


def truth_action_to_category(action: str) -> str | None:
    if action in UNSCOREABLE_TRUTH_ACTIONS:
        return None
    return TRUTH_ACTION_TO_CATEGORY.get(action)


def llm_judge(case_summary: str, classifier_output: dict, ground_truth: dict, client) -> dict:
    dev_action = ground_truth.get("developer_action", "UNKNOWN")

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

    gt_details = ground_truth.get("developer_action_reasoning", ground_truth.get("reasoning", ""))
    follow_ups = ground_truth.get("follow_up_commits", [])
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
        from src.apa.llm_config import get_provider
        provider = get_provider()
        if provider == "deepseek":
            model = "deepseek-chat"
        elif provider == "openrouter":
            model = "openai/gpt-4o"
        else:
            model = "gpt-4o"

        from src.apa.llm_usage import record_usage
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
        record_usage(response, model, call_type="chat", label="judge_rpa_vs_apa.judge")
        import json
        data = json.loads(response.choices[0].message.content)
        return {"verdict": data.get("verdict", "EVALUATION_ERROR"), "reasoning": data.get("reasoning", "")}
    except Exception as e:
        return {"verdict": "EVALUATION_ERROR", "reasoning": f"{type(e).__name__}: {e}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge RPA and APA outputs against ground truth for an existing batch_results.json."
    )
    parser.add_argument("batch_dir", type=Path, help="Directory containing batch_results.json")
    parser.add_argument(
        "--truth-path",
        type=Path,
        default=None,
        help=(
            "Optional frozen truth_eval_results.json (or compatible JSON) to reuse. "
            "If provided, the script will NOT scrape ground truth."
        ),
    )
    return parser.parse_args()


def _load_frozen_truth(truth_path: Path) -> dict[str, dict]:
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    # Supported formats:
    # 1) list[ {run_id, ground_truth:{developer_action,...,ground_truth_category}} ]
    # 2) dict with key "cases" holding that list
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        items = payload.get("cases")
    elif isinstance(payload, list):
        items = payload
    else:
        raise SystemExit(f"Unsupported truth file format: {truth_path}")

    out: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id")
        gt = item.get("ground_truth")
        if not run_id or not isinstance(gt, dict):
            continue
        action = (gt.get("developer_action") or "").strip()
        gt_category = gt.get("ground_truth_category")
        if gt_category is None:
            gt_category = truth_action_to_category(action)
        out[str(run_id)] = {
            "developer_action": action,
            "reasoning": gt.get("reasoning") or "",
            "method": gt.get("method") or gt.get("classification_method") or "frozen",
            "ground_truth_category": gt_category,
        }

    if not out:
        raise SystemExit(f"No usable ground truth entries found in: {truth_path}")
    return out


def load_runs(run_ids: set[str]) -> dict[str, dict]:
    runs = {}
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = raw.get("_id")
            if rid in run_ids:
                runs[rid] = raw
                if len(runs) == len(run_ids):
                    break
    return runs


def summarize_case(event) -> str:
    return (
        f"Repo: {event.repo}  Branch: {event.branch} (protected: {event.is_protected_branch})\n"
        f"Commit: {event.commit_title}\n"
        f"Workflow: {event.workflow}\n"
        f"Failed jobs: {event.failed_jobs_count}/{event.n_jobs}\n"
        f"Failure detection: {event.failure_detection}"
    )


def maybe_upgrade_pr_merged(gt, repo: str) -> None:
    if gt.developer_action != "PR_MERGED":
        return
    import re

    m = re.search(r"PR #(\d+)", gt.developer_action_reasoning or "")
    parts = repo.split("/")
    if not m or len(parts) != 2:
        return

    pr_num = int(m.group(1))
    pr_files = fetch_pr_files(parts[0], parts[1], pr_num)
    if not pr_files:
        return

    action, reasoning, signals = classify_pr_fix_type(
        {"title": gt.developer_action_reasoning or "", "body": ""},
        pr_files,
    )
    gt.developer_action = action
    gt.developer_action_reasoning = f"[pr_files] {reasoning}"
    gt.classification_method = "pr_files"


def main() -> None:
    from src.apa.llm_config import make_client
    client = make_client()
    
    args = parse_args()
    batch_path = args.batch_dir / "batch_results.json"
    truth_path = args.batch_dir / "truth_eval_results.json"
    output_path = args.batch_dir / "rpa_apa_judged_results.json"
    summary_path = args.batch_dir / "rpa_apa_judged_summary.json"

    with open(batch_path, "r", encoding="utf-8") as f:
        batch = json.load(f)
    cases = batch["cases"]

    run_ids = {case["run_id"] for case in cases}
    print(f"Loading {len(run_ids)} runs from {RUNS_PATH}...")
    runs_by_id = load_runs(run_ids)
    print(f"Found {len(runs_by_id)}/{len(run_ids)} runs.\n")

    frozen_truth_by_run_id: dict[str, dict] | None = None
    if args.truth_path:
        frozen_truth_by_run_id = _load_frozen_truth(args.truth_path.expanduser())
        missing = sorted(rid for rid in run_ids if rid not in frozen_truth_by_run_id)
        if missing:
            raise SystemExit(
                f"Frozen truth file is missing {len(missing)} run_ids. First missing: {missing[0]}"
            )
        print(f"Using frozen truth from: {args.truth_path} (no scraping)\n")

    results = []
    gt_results = []
    rpa_verdicts = Counter()
    apa_verdicts = Counter()
    gt_actions = Counter()
    gt_categories = Counter()
    rpa_categories = Counter()
    apa_categories = Counter()

    for i, case in enumerate(cases, 1):
        run_id = case["run_id"]
        repo = case.get("repository_name") or case.get("repo") or ""
        print(f"[{i:>3}/{len(cases)}] {repo}")

        raw = runs_by_id.get(run_id)
        if not raw:
            print("         ↳ run not found")
            continue

        try:
            event = intake(raw)
            if frozen_truth_by_run_id is not None:
                gt_dict = frozen_truth_by_run_id[str(run_id)]
                gt_results.append(
                    {
                        "index": i,
                        "run_id": run_id,
                        "label": repo,
                        "ground_truth": gt_dict,
                    }
                )
                truth_path.write_text(json.dumps(gt_results, indent=2, ensure_ascii=False), encoding="utf-8")
                gt_actions[gt_dict.get("developer_action") or ""] += 1
                if gt_dict.get("ground_truth_category"):
                    gt_categories[gt_dict["ground_truth_category"]] += 1
                print(
                    "         ↳ ground truth: "
                    f"{gt_dict.get('developer_action')} ({gt_dict.get('method')}) [frozen]"
                )
            else:
                gt = scrape_ground_truth(event)
                maybe_upgrade_pr_merged(gt, repo)
                gt_dict = {
                    "developer_action": gt.developer_action,
                    "reasoning": gt.developer_action_reasoning,
                    "method": gt.classification_method,
                    "ground_truth_category": truth_action_to_category(gt.developer_action),
                }
                gt_results.append(
                    {
                        "index": i,
                        "run_id": run_id,
                        "label": repo,
                        "ground_truth": gt_dict,
                    }
                )
                truth_path.write_text(json.dumps(gt_results, indent=2, ensure_ascii=False), encoding="utf-8")
                gt_actions[gt.developer_action] += 1
                if gt_dict["ground_truth_category"]:
                    gt_categories[gt_dict["ground_truth_category"]] += 1
                print(f"         ↳ ground truth: {gt.developer_action} ({gt.classification_method})")
        except Exception as e:
            print(f"         ↳ ground truth error: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        summary = summarize_case(event)
        rpa_classifier = (case.get("rpa") or {}).get("classification") or {}
        apa_classifier = (case.get("apa") or {}).get("classification") or {}

        gt_category = gt_dict.get("ground_truth_category")
        rpa_pred_cat = rpa_classifier.get("category")
        apa_pred_cat = apa_classifier.get("category")
        rpa_judge = llm_judge(summary, rpa_classifier, gt_dict, client)
        apa_judge = llm_judge(summary, apa_classifier, gt_dict, client)

        if rpa_pred_cat:
            rpa_categories[rpa_pred_cat] += 1
        if apa_pred_cat:
            apa_categories[apa_pred_cat] += 1

        rpa_verdicts[rpa_judge["verdict"]] += 1
        apa_verdicts[apa_judge["verdict"]] += 1

        print(f"         ↳ RPA judge: {rpa_judge['verdict']}")
        print(f"         ↳ APA judge: {apa_judge['verdict']}")

        results.append(
            {
                "index": i,
                "run_id": run_id,
                "repo": repo,
                "ground_truth": gt_dict,
                "rpa": {
                    "classification": rpa_classifier,
                    "case_summary": summary,
                    "judge": rpa_judge,
                },
                "apa": {
                    "classification": apa_classifier,
                    "case_summary": summary,
                    "judge": apa_judge,
                },
            }
        )
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "n_cases": len(results),
        "ground_truth_actions": dict(gt_actions),
        "ground_truth_categories": dict(gt_categories),
        "rpa_predicted_categories": dict(rpa_categories),
        "apa_predicted_categories": dict(apa_categories),
        "rpa_verdicts": dict(rpa_verdicts),
        "apa_verdicts": dict(apa_verdicts),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 64}")
    print(f"DONE: {len(results)} cases")
    print("RPA verdicts:")
    for k, v in rpa_verdicts.items():
        print(f"  {k}: {v}")
    print("APA verdicts:")
    for k, v in apa_verdicts.items():
        print(f"  {k}: {v}")
    print(f"\nTruth file: {truth_path}")
    print(f"Judged file: {output_path}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
