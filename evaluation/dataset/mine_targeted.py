# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Targeted corpus top-up: mine NEW cases for under-represented categories only.
#   DEPENDENCY  (GT actions PIN_VERSION, DEPENDENCY_CHANGE)
#   CONFIG      (GT action WORKFLOW_FIX)
# Reuses build_dataset's exact inclusion gates (context, ground-truth, informative-log,
# one-per-repo) but (a) restricts the GT gate to the target actions so only the wanted
# categories pass, and (b) skips run_ids/repos already in the existing corpus so we add
# genuinely new, diverse cases. Logs are range-fetched from Zenodo via ZIP_URL.
import argparse, gzip, json, os, sys, traceback
from pathlib import Path
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

import build_dataset as B
from src.apa.intake_parser import intake
from src.apa.log_extractor import extract_log_excerpt

ACTION_BUCKET = {
    "PIN_VERSION": "DEPENDENCY", "DEPENDENCY_CHANGE": "DEPENDENCY",
    "WORKFLOW_FIX": "CONFIG", "CODE_FIX": "CODE", "CODE_CHANGE": "CODE",
}

def load_existing_run_ids(paths):
    seen_runs, seen_repos = set(), set()
    for p in paths:
        if not os.path.exists(p):
            continue
        op = gzip.open if p.endswith(".gz") else open
        for l in op(p, "rt", encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            rid = (r.get("intake") or r).get("run_id") or r.get("run_id")
            repo = (r.get("intake") or r).get("repo") or r.get("repo")
            if rid: seen_runs.add(str(rid))
            if repo: seen_repos.add(repo)
    return seen_runs, seen_repos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-path", default="data/runs_zenodo.json.gz")
    ap.add_argument("--out", default="data/dataset_topup.jsonl.gz")
    ap.add_argument("--need-dependency", type=int, default=35)
    ap.add_argument("--need-config", type=int, default=25)
    ap.add_argument("--scan-start", type=int, default=1)
    ap.add_argument("--scan-end", type=int, default=600_000)
    ap.add_argument("--progress-every", type=int, default=5_000)
    args = ap.parse_args()

    assert os.environ.get("ZIP_URL"), "ZIP_URL not set — needed to range-fetch logs"
    zip_path = os.environ["ZIP_URL"]

    existing_runs, existing_repos = load_existing_run_ids([
        "data/eval_big100.jsonl",
        "data/dataset_remote_250.jsonl.gz",
        "data/dataset_remote_120.jsonl.gz",
        "data/dataset_remote_next.jsonl.gz",
    ])
    print(f"existing: {len(existing_runs)} run_ids, {len(existing_repos)} repos to avoid")

    target_actions = {"PIN_VERSION", "DEPENDENCY_CHANGE", "WORKFLOW_FIX"}
    need = {"DEPENDENCY": args.need_dependency, "CONFIG": args.need_config}
    got = {"DEPENDENCY": 0, "CONFIG": 0}

    writer = B._CaseWriter(Path(args.out), "jsonl.gz")
    from archive.ground_truth_scraper import quick_gt_preflight, scrape_ground_truth
    from evals.coarse_eval import substantive_fix_buckets

    repo_counts = {}
    scanned = ctx = gtp = uninf = sel = 0
    print(f"target: +{need['DEPENDENCY']} DEPENDENCY, +{need['CONFIG']} CONFIG")

    with gzip.open(args.runs_path, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned < args.scan_start: continue
            if scanned > args.scan_end: break
            if args.progress_every and scanned % args.progress_every == 0:
                print(f"  scanned {scanned:,} | ctx {ctx} | gt_pass {gtp} | "
                      f"DEP {got['DEPENDENCY']}/{need['DEPENDENCY']} CONFIG {got['CONFIG']}/{need['CONFIG']}")
            if got["DEPENDENCY"] >= need["DEPENDENCY"] and got["CONFIG"] >= need["CONFIG"]:
                break
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not B.passes_context_gate(run, protected_only=True, min_steps_per_job=2, require_shell_step=True):
                continue
            ctx += 1
            repo_full = run.get("repository_name", "")
            meta = run.get("metadata") or {}
            branch = meta.get("head_branch", "")
            started_at = B._run_started_at(meta)
            sha = (meta.get("head_commit") or {}).get("id", "")
            owner, repo_name = B._parse_owner_repo(repo_full)
            if not owner or not repo_name or not sha or not branch or not started_at:
                continue
            run_id = str((meta.get("id") or run.get("id") or ""))
            # skip already-used run_ids and repos (diversity)
            if run_id in existing_runs or repo_full in existing_repos:
                continue
            if repo_counts.get(repo_full, 0) >= 1:
                continue
            # Mark this repo as touched NOW so a rejected scrape doesn't get re-scraped
            # every time the same repo reappears in the scan (one shot per repo).
            repo_counts[repo_full] = 1
            # Cheap preflight only to drop AMBIGUOUS (genuinely unscorable). Everything
            # else gets a full scrape — NO_IMMEDIATE_RESPONSE cases often still have a
            # delayed fix the full scrape finds, so skipping them (88% of the pool) starved
            # the miner of targets. PIN_VERSION/WORKFLOW_FIX from preflight are kept too.
            try:
                pf = quick_gt_preflight(owner, repo_name, sha, branch, started_at)
            except Exception:
                continue
            if pf == "AMBIGUOUS":
                continue
            # Full ground-truth scrape — produces the fine developer_action + reasoning
            # the eval buckets on (PIN_VERSION / WORKFLOW_FIX / ...).
            try:
                event = intake(run)
                gt = scrape_ground_truth(event)
            except Exception:
                continue
            gt_action = getattr(gt, "developer_action", "UNKNOWN")
            gt_reason = getattr(gt, "developer_action_reasoning", "") or ""
            gt_method = getattr(gt, "classification_method", "none")
            # Quality gate: only accept the SAME high-confidence GT methods used by the
            # existing 253-case corpus (pr_files / llm / rubric). Drop unclear/heuristic
            # methods so the topped-up cases are no noisier than the originals.
            if gt_method not in ("pr_files", "llm", "rubric"):
                continue
            # Bucket via the SAME logic the eval uses (substantive fix buckets on reasoning).
            prim, _bset = substantive_fix_buckets(gt_reason)
            bucket = prim or ACTION_BUCKET.get(gt_action)
            if bucket not in need or got.get(bucket, 0) >= need[bucket]:
                continue
            gtp += 1
            try:
                tarball = B.tarball_path(run)
                excerpts, sample_errors = [], []
                for fs in event.failed_steps[:3]:
                    ex = extract_log_excerpt(zip_path=zip_path, tarball_name=tarball,
                                             job_file=fs.job_file, step_label=fs.step_label or "")
                    excerpts.append(ex)
                    for ml in ex.error_marker_lines[:2]:
                        c = ml.strip()[:120]
                        if c and c not in sample_errors: sample_errors.append(c)
                combined = "\n".join(ex.as_prompt_text() for ex in excerpts)
                if not B.is_informative_log(combined):
                    repo_counts[repo_full] = 0; uninf += 1
                    continue
                payload = []
                for ex in excerpts:
                    text = ex.as_prompt_text()
                    text = B._redact_secrets(text)
                    if len(text) > 20000: text = text[:19999] + "…"
                    payload.append({"job_file": getattr(ex,"job_file",""),
                                    "step_label": getattr(ex,"step_label",""),
                                    "strategy": getattr(ex,"strategy_used",""), "text": text})
                case = {
                    "case_label": f"{event.repo} — {event.commit_title[:60]}",
                    "preflight_action": pf,
                    "target_bucket": bucket,
                    "mined_topup": True,
                    "gt_fix_window_days": int(os.environ.get("GT_FIX_WINDOW_DAYS", "7")),
                    "ground_truth": {"action": gt_action, "method": gt_method, "reasoning": gt_reason},
                    "intake": {"run_id": event.run_id, "repo": event.repo, "workflow": event.workflow,
                        "branch": event.branch, "event": event.event,
                        "is_protected_branch": event.is_protected_branch, "commit_sha": event.commit_sha,
                        "commit_title": event.commit_title, "conclusion": event.conclusion,
                        "failure_detection": event.failure_detection,
                        "failed_jobs_count": event.failed_jobs_count, "n_jobs": event.n_jobs,
                        "all_failures_are_tooling_artifacts": event.all_failures_are_tooling_artifacts},
                    "extraction": {"total_steps_extracted": len(excerpts),
                        "strategies": [ex.strategy_used for ex in excerpts],
                        "error_markers_found": sum(len(ex.error_marker_lines) for ex in excerpts),
                        "sample_error_lines": sample_errors[:5], "log_excerpts": payload},
                }
                writer.write(case); sel += 1; got[bucket] += 1
                print(f"  ✓ [{bucket}] {got[bucket]}/{need[bucket]}  {repo_full}  action={gt_action}")
            except Exception as e:
                repo_counts[repo_full] = 0
                print(f"  ! build failed for {repo_full}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=1)

    writer.close()
    print(f"\nDone. Selected {sel} ({got}) from {scanned:,} scanned. "
          f"ctx={ctx} gt_pass={gtp} uninformative={uninf}")

if __name__ == "__main__":
    main()
