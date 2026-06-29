#!/usr/bin/env python3
"""
run_eval_500.py
===============
RPA-vs-APA evaluation over the diverse-500 dataset, via OpenRouter, with a
hard budget cap and full checkpoint/resume.

Design choices baked in (decided with the user):
  - Provider: OpenRouter (LLM_PROVIDER=openrouter, needs OPENROUTER_API_KEY).
  - Cheap nodes (planner, per-tool likelihoods, deep-log): deepseek/deepseek-chat
  - Reasoning nodes (classify, devil's advocate, action, judge): openai/gpt-4o-mini
  - Cost: read OpenRouter's REAL per-call cost (usage.cost); session total is
    exact, not estimated.
  - Budget: hard cap (default $1). Checked BETWEEN cases, so an abort happens at
    a clean checkpoint with all completed cases already saved.
  - Resume: every completed case is written immediately to OUTPUT_PATH; a
    re-run skips cases already present (keyed by run_id), so stopping for any
    reason (budget, crash, Ctrl-C) and re-running continues where it left off.

Usage:
  export OPENROUTER_API_KEY=...      # (also GITHUB_TOKEN for ground truth)
  python run_eval_500.py --n 20 --budget 1.0
  python run_eval_500.py --n 30 --budget 1.0     # resumes; runs 10 more, etc.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import traceback
from pathlib import Path

# ── provider + model tiers MUST be set before importing APA modules ──
# Cost-tiered, all-DeepSeek system + independent OpenAI judge:
#   - cheap nodes (planner, per-tool likelihoods, deep-log): deepseek-chat
#   - classify (the actual diagnosis):                         deepseek-reasoner
#   - devil's advocate + recommended action (secondary):       deepseek-chat
#   - the two judges:                                          OpenAI gpt-4o-mini
# This is ONE reasoner call per case (classify only) instead of five.
# The judge runs on a different provider so the evaluator is independent of the
# system under test. DeepSeek gives no per-call cost, so DeepSeek spend is
# ESTIMATED; OpenAI judge spend is estimated too (both via the price table).
os.environ.setdefault("LLM_PROVIDER", "deepseek")
os.environ.setdefault("CI_AGENT_PLANNER_MODE", "hybrid")
os.environ["CI_AGENT_MODEL"] = os.environ.get("CI_AGENT_MODEL", "deepseek-chat")
os.environ["CI_AGENT_CLASSIFY_MODEL"] = os.environ.get("CI_AGENT_CLASSIFY_MODEL", "deepseek-reasoner")
os.environ["CI_AGENT_SECONDARY_MODEL"] = os.environ.get("CI_AGENT_SECONDARY_MODEL", "deepseek-chat")
os.environ["JUDGE_PROVIDER"] = os.environ.get("JUDGE_PROVIDER", "openai")
os.environ["JUDGE_MODEL"] = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

# ── cost controls (DeepSeek spend) ──────────────────────────────────────────
# Eval scores only the category, so skip the recommended-action LLM call, and
# cap investigation steps (fewer planner + per-tool likelihood calls). Override
# via env if you want the full agent. CI_AGENT_CLASSIFY_MODEL stays on the
# reasoner unless you set it to deepseek-chat for a bigger (quality-risky) cut.
os.environ.setdefault("CI_AGENT_SKIP_ACTION", "1")
os.environ.setdefault("CI_AGENT_MAX_STEPS", "3")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.apa.intake_parser import RunEvent
from src.apa.agent import (
    build_agent_graph, _initial_tools_for_event, _fetch_commit_diff,
    _summarize_patch_files, _fetch_run_history, _failed_step_summary,
)
from src.apa.bayesian_tracker import BeliefState
from src.apa.bayesian_tracker_dual import DualTracker
from src.apa.llm_config import make_client
from src.apa.llm_usage import get_session_cost, check_budget, BudgetExceeded
from archive.ground_truth_scraper import scrape_ground_truth
from evals.evaluation_judge import llm_judge, build_case_summary

DATA = Path("data/dataset_diverse_500.jsonl.gz")
OUTPUT = Path("data/eval_500_results.jsonl")     # one JSON object per line (resumable)
NOT_SCORABLE = {
    "UNRELATED", "PR_CLOSED_UNMERGED", "PR_MERGED", "PR_MERGED_NO_FILES",
    "PR_MERGED_UNCLEAR", "REVERT", "EVALUATION_ERROR", "UNKNOWN",
    "NO_IMMEDIATE_RESPONSE", "NO_RELEVANT_RESPONSE", "NO_FOLLOW_UP",
}


def load_cases():
    out = []
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def done_run_ids() -> set:
    """run_ids already completed (for resume)."""
    if not OUTPUT.exists():
        return set()
    ids = set()
    for line in OUTPUT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line).get("run_id", ""))
        except Exception:
            pass
    return ids


def _event_from_case(c: dict) -> RunEvent:
    ik = c.get("intake", {})
    md = c.get("metadata", {})
    # run_number / attempt: prefer explicit fields; otherwise parse the trailing
    # "_<run_number>_<attempt>" off the run_id (e.g. "...tests.yml_740_1").
    # The run-history prefetch needs run_number, and _get_event keys its cache
    # on (run_id, repo, run_number, attempt).
    run_number = ik.get("run_number") or md.get("run_number")
    attempt = ik.get("attempt") or md.get("attempt")
    if run_number is None:
        rid = ik.get("run_id", "")
        m = __import__("re").search(r"_(\d+)_(\d+)$", rid)
        if m:
            run_number = int(m.group(1))
            attempt = attempt or int(m.group(2))
    attempt = attempt or 1

    # The 500-dataset stores none of the fields the RPA signal battery gates on
    # (available_signals, is_protected_branch, a real failure_detection), so the
    # RPA baseline would otherwise collapse to the informed prior. Recompute
    # them here from the data we do have, matching the GitHub adapter's logic.
    branch = ik.get("branch", "")
    _PROTECTED = ("main", "master", "release", "releases/", "prod", "production")
    bl = (branch or "").lower()
    is_protected = any(bl == p or bl.startswith(p) for p in _PROTECTED)

    detection = ik.get("failure_detection", "")
    if not detection or detection == "not_a_failure":
        # The dataset lost the step-level mode; infer the weakest sensible
        # default so detection_mode still discriminates instead of being inert.
        detection = "job_level_fallback"

    # Full signal set so _build_preprocessing_state actually runs the battery.
    available = [
        "error_text", "many_jobs_failed", "branch_type",
        "commit_message", "previous_runs", "parent_commit_run", "detection_mode",
    ]

    return RunEvent(
        source="github", run_id=ik.get("run_id", ""), repo=ik.get("repo", ""),
        event=ik.get("event", ""), conclusion=ik.get("conclusion", "failure"),
        started_at=ik.get("started_at", ""), n_jobs=ik.get("n_jobs", 0) or 1,
        failed_jobs_count=ik.get("failed_jobs_count", 0) or 1, duration_sec=None,
        branch=branch, commit_sha=ik.get("commit_sha", ""),
        commit_title=ik.get("commit_title", ""),
        commit_author=ik.get("commit_author", ""), workflow=ik.get("workflow", ""),
        run_number=run_number, attempt=attempt,
        is_protected_branch=is_protected, failure_detection=detection,
        available_signals=available,
    )


def run_case(c: dict, client, gt=None) -> dict:
    from dataclasses import asdict
    ik = c.get("intake", {})
    event = _event_from_case(c)
    # The agent re-parses state["run_event"] back into a RunEvent, so it must
    # contain ALL RunEvent fields (incl. non-default source/started_at/
    # duration_sec). Use the fully-populated dataclass dict, not the raw intake.
    kwargs = asdict(event)
    extraction = c.get("extraction", {})
    excerpts = extraction.get("log_excerpts", [])
    log_texts = [ex.get("text", "") for ex in excerpts if isinstance(ex, dict) and "text" in ex]
    error_lines = extraction.get("sample_error_lines", [])

    # 1. RPA (deterministic, no LLM) — run the REAL 9-signal preprocessing
    # battery (_build_preprocessing_state), not a stripped 4-signal tracker, so
    # this is the actual RPA baseline. Feed it the case's extracted log evidence
    # via precomputed_log_evidence so error_text / mentioned_files signals fire.
    from src.apa.agent import _build_preprocessing_state
    raw_for_rpa = {"precomputed_log_evidence": {
        "error_lines": error_lines,
        "mentioned_files": extraction.get("mentioned_files", []),
        "log_excerpt_texts": log_texts,
    }}
    pp = _build_preprocessing_state(event, raw_for_rpa)
    _ps = pp["preprocessing_summary"]
    rpa_result = {
        "category": _ps["top_category"],
        "confidence": _ps["top_probability"],
        "all_probabilities": pp["beliefs"],
        "signals_applied": _ps["signals_applied"],
        "mode": "rpa",
    }

    # 2. Ground truth (relevance-filtered) — cheap, lets us skip before APA.
    #    May be pre-scraped by a balanced selector and passed in to avoid a
    #    second scrape.
    if gt is None:
        gt = scrape_ground_truth(event)
    dev_action = getattr(gt, "developer_action", "UNKNOWN")
    # Feed the judge the SAME extracted log the agent saw, not just the generic
    # "exit code 1" markers. Without this the judge auto-marks every
    # log-dependent category WRONG for lack of evidence (see build_case_summary).
    judge_log = "\n\n".join(t for t in log_texts if t)
    case_summary = build_case_summary(event, error_lines, log_excerpt=judge_log)

    if dev_action in NOT_SCORABLE:
        ns = {"verdict": "NOT_SCORABLE",
              "reasoning": f"Ground truth action is {dev_action}.",
              "confidence": 0.0, "evidence_used": [f"dev_action={dev_action}"]}
        return {
            "run_id": event.run_id, "repo": event.repo,
            "commit": event.commit_title[:60],
            "ground_truth": {"action": dev_action,
                             "method": gt.classification_method,
                             "reasoning": gt.developer_action_reasoning},
            "rpa": {"prediction": rpa_result, "judge": ns},
            "apa": {"prediction": {}, "judge": ns},
        }

    # 3. APA agent (seeded from informed prior, NOT RPA posterior)
    changed_files = []
    commit_diff_summary = {}
    if event.repo and event.commit_sha:
        changed_files = _fetch_commit_diff(event.repo, event.commit_sha).get("files", [])
        commit_diff_summary = _summarize_patch_files(changed_files)
    run_history = {}
    if event.repo and event.branch and event.run_number:
        try:
            run_history = _fetch_run_history(event.repo, event.branch, event.workflow or "", event.run_number)
        except Exception:
            pass

    prior_bs = BeliefState()
    initial_state = {
        "run_event": kwargs, "raw_run": c,
        "beliefs": dict(prior_bs.probabilities),
        "belief_history": list(prior_bs.history),
        "confidence": prior_bs.confidence(), "entropy": prior_bs.entropy(),
        "tools_available": _initial_tools_for_event(c, event),
        "tools_called": [], "investigation_log": [], "current_step": 0, "done": False,
        "error_lines": error_lines,
        "mentioned_files": extraction.get("mentioned_files", []),
        "log_excerpt_texts": log_texts,
        "changed_files": changed_files, "commit_diff": commit_diff_summary,
        "failed_step_context": _failed_step_summary(event),
        "dependency_changes": {}, "run_history": run_history,
        "similar_failures": [], "workflow_contents": [], "runner_environment": {},
        "pr_context": {}, "semantic_diff_links": [],
        "preprocessing_summary": {"top_category": rpa_result["category"]},
        "classification": {}, "api_key": "",
        "model": os.environ["CI_AGENT_MODEL"], "_next_tool": "",
    }
    apa_state = build_agent_graph().invoke(initial_state)
    apa_result = apa_state.get("classification", {})

    # 4. Judge both
    rpa_judge = llm_judge(case_summary, rpa_result, gt, client)
    apa_judge = llm_judge(case_summary, apa_result, gt, client)

    return {
        "run_id": event.run_id, "repo": event.repo,
        "commit": event.commit_title[:60], "error_lines": error_lines[:8],
        "ground_truth": {"action": gt.developer_action,
                         "method": gt.classification_method,
                         "reasoning": gt.developer_action_reasoning},
        "rpa": {"prediction": rpa_result, "judge": rpa_judge},
        "apa": {"prediction": apa_result, "judge": apa_judge},
    }


def append_result(rec: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize():
    if not OUTPUT.exists():
        print("no results yet"); return
    from collections import Counter
    rpa_c = apa_c = apa_p = scor = 0
    gt_dist = Counter()
    # Divergence analysis: when RPA and APA DISAGREE, who is right? This is the
    # number that actually measures APA's added value over the baseline.
    div_total = div_apa_right = div_rpa_right = div_both_wrong = 0
    for line in OUTPUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rv = r["rpa"]["judge"]["verdict"]; av = r["apa"]["judge"]["verdict"]
        if rv in ("NOT_SCORABLE", "EVALUATION_ERROR", "") or av in ("NOT_SCORABLE", "EVALUATION_ERROR", ""):
            continue
        scor += 1
        gt_dist[r["ground_truth"]["action"]] += 1
        rpa_c += rv == "CORRECT"; apa_c += av == "CORRECT"; apa_p += av == "PARTIAL"
        rp = r["rpa"]["prediction"].get("category") if isinstance(r["rpa"]["prediction"], dict) else None
        ap = r["apa"]["prediction"].get("category") if isinstance(r["apa"]["prediction"], dict) else None
        if rp != ap:   # they disagreed
            div_total += 1
            if av == "CORRECT" and rv != "CORRECT": div_apa_right += 1
            elif rv == "CORRECT" and av != "CORRECT": div_rpa_right += 1
            else: div_both_wrong += 1
    print(f"\n=== SUMMARY ===  scorable={scor}")
    if scor:
        print(f"RPA accuracy:            {100*rpa_c/scor:.1f}%  ({rpa_c}/{scor})")
        print(f"APA accuracy:            {100*apa_c/scor:.1f}%  ({apa_c}/{scor})")
        print(f"APA accuracy (+partial): {100*(apa_c+apa_p)/scor:.1f}%")
        print(f"\n-- when RPA and APA DISAGREE (n={div_total}) --")
        print(f"  APA right, RPA wrong:  {div_apa_right}   <- APA's added value")
        print(f"  RPA right, APA wrong:  {div_rpa_right}   <- APA's regressions")
        print(f"  both wrong/other:      {div_both_wrong}")
        print(f"\n-- ground-truth class mix (watch for CODE_FIX skew) --")
        for k, v in gt_dist.most_common():
            print(f"  {k:20} {v}")
    print(f"\nSession spend so far: ${get_session_cost():.4f}")


def run_balanced(args) -> int:
    """Collect ~equal numbers of cases per COARSE ground-truth bucket.

    Scrapes GT first (cheap) and runs the expensive APA agent ONLY when the
    case's bucket still has room, so budget is spent filling the corpus, not on
    over-represented CODE cases. The rare buckets (CONFIG, then DEPENDENCY) are
    sought first; TRANSIENT is effectively absent from this corpus (no
    retry-fixed cases) and will simply stay empty — reported honestly.
    """
    from evals.coarse_eval import gt_bucket, score, load_results
    from build_dataset import is_informative_log

    if os.environ["LLM_PROVIDER"] == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set."); return 1
    os.environ.setdefault("LLM_USAGE_LOG_PATH", "data/eval_balanced_usage.jsonl")
    client = make_client()

    import collections
    per_bucket = args.balanced
    targets = {"CODE": per_bucket, "DEPENDENCY": per_bucket, "CONFIG": per_bucket, "TRANSIENT": per_bucket}
    filled = collections.Counter()

    cases = load_cases()
    done = done_run_ids()
    remaining = [c for c in cases if c.get("intake", {}).get("run_id", "") not in done]
    # Already-saved cases count toward quotas so a resumed run tops up correctly.
    for r in (load_results([str(OUTPUT)]) if OUTPUT.exists() else []):
        b = gt_bucket(r.get("ground_truth", {}).get("action"))
        if b:
            filled[b] += 1

    # Seek rare buckets first: PR_MERGED (only source of CONFIG) -> PIN_VERSION
    # (DEPENDENCY) -> CODE_FIX (CODE). REVERT can never be scorable, skip it.
    pref_order = {"PR_MERGED": 0, "PIN_VERSION": 1, "CODE_FIX": 2}
    remaining = [c for c in remaining if c.get("preflight_action") != "REVERT"]
    remaining.sort(key=lambda c: pref_order.get(c.get("preflight_action"), 9))

    print(f"Provider: {os.environ['LLM_PROVIDER']} | reason={os.environ['CI_AGENT_CLASSIFY_MODEL']}")
    print(f"BALANCED mode -> {OUTPUT}  | target {per_bucket}/bucket | budget ${args.budget}")
    print(f"already have: {dict(filled)}")

    max_cases = int(getattr(args, "max_cases", 0) or 0)
    processed = scraped = 0
    for c in remaining:
        if all(filled[b] >= targets[b] for b in ("CODE", "DEPENDENCY", "CONFIG")):
            print("\nAll reachable buckets full. Done."); break
        if max_cases and sum(filled.values()) >= max_cases:
            print(f"\nReached --max-cases {max_cases}. Done."); break
        try:
            check_budget(args.budget)
        except BudgetExceeded as e:
            print(f"\n[budget] {e}. Stopping cleanly; {processed} cases this run."); break

        # Informative-log gate: skip cases whose log carries no real diagnostic
        # (only "exit code 1" + activity noise). These are unsolvable from the
        # evidence, so they only add base-rate collapse. Cheap local check before
        # the expensive scrape.
        _log = "\n".join(e.get("text", "") for e in (c.get("extraction", {}).get("log_excerpts") or []))
        if not is_informative_log(_log):
            continue

        event = _event_from_case(c)
        try:
            gt = scrape_ground_truth(event)
        except Exception as e:
            print(f"  scrape failed {event.repo}: {e}"); continue
        scraped += 1
        b = gt_bucket(getattr(gt, "developer_action", None))
        if b is None or filled[b] >= targets[b]:
            continue  # not scorable, or this bucket already full -> skip the agent

        print(f"\n[{b} {filled[b]+1}/{targets[b]}] {event.repo}  (scraped={scraped}, spend ${get_session_cost():.4f})")
        try:
            rec = run_case(c, client, gt=gt)
            append_result(rec)
            filled[b] += 1; processed += 1
            print(f"    rpa={rec['rpa']['judge']['verdict']} apa={rec['apa']['judge']['verdict']} "
                  f"gt={rec['ground_truth']['action']} bucket={b}")
        except BudgetExceeded as e:
            print(f"\n[budget] hit mid-case: {e}. case not saved."); break
        except Exception as e:
            print(f"    ERROR: {e}"); traceback.print_exc(limit=2)

    print(f"\nScraped {scraped} candidates, ran agent on {processed}. Buckets: {dict(filled)}")
    print(f"Session spend: ${get_session_cost():.4f}")
    print("\n========== COARSE (bucket-match, judge-free) ==========")
    score(load_results([str(OUTPUT)]))
    print("\n========== COARSE (balanced equal-n) ==========")
    score(load_results([str(OUTPUT)]), balanced=True)
    return 0


def main():
    global OUTPUT, DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="process up to N not-yet-done cases")
    ap.add_argument("--budget", type=float, default=1.0, help="hard USD cap; abort cleanly when reached")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--balanced", type=int, default=0, metavar="PER_BUCKET",
                    help="balanced mode: collect this many cases PER COARSE GT bucket "
                         "(CODE/DEPENDENCY/CONFIG/TRANSIENT). Scrapes GT first and "
                         "skips the expensive agent on already-full buckets.")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="balanced mode: hard stop once this many cases are saved "
                         "(prevents over-scraping when a rare bucket can't fill).")
    ap.add_argument("--dataset", type=str, default="",
                    help="override input dataset path (e.g. dataset_diverse_200.jsonl.gz)")
    ap.add_argument("--output", type=str, default="",
                    help="override results file (balanced mode defaults to "
                         "data/eval_balanced_results.jsonl)")
    args = ap.parse_args()

    if args.dataset:
        DATA = Path(args.dataset)
    if args.output:
        OUTPUT = Path(args.output)
    elif args.balanced:
        OUTPUT = Path("data/eval_balanced_results.jsonl")

    if args.summarize:
        summarize()
        try:
            from evals.coarse_eval import score, load_results
            print("\n========== COARSE (bucket-match, judge-free) ==========")
            score(load_results([str(OUTPUT)]))
        except Exception as e:
            print(f"(coarse summary skipped: {e})")
        return 0

    if args.balanced:
        return run_balanced(args)

    if os.environ["LLM_PROVIDER"] == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set."); return 1

    os.environ.setdefault("LLM_USAGE_LOG_PATH", "data/eval_500_usage.jsonl")
    client = make_client()
    cases = load_cases()
    done = done_run_ids()
    remaining = [c for c in cases if c.get("intake", {}).get("run_id", "") not in done]

    # Skip preflight labels that the judge ALWAYS drops as NOT_SCORABLE no matter
    # what the scrape finds — only REVERT (a revert never pins the failure
    # category). Running these wastes scrape + agent budget on unscorable cases.
    # NOTE: PR_MERGED is intentionally NOT skipped here even though it is in
    # NOT_SCORABLE: a preflight PR_MERGED case gets RE-CLASSIFIED at scrape time
    # (classify_pr_fix_type) into a concrete WORKFLOW_FIX / CODE_FIX / PIN_VERSION
    # label, so most of these ARE scorable — and they supply the WORKFLOW_FIX
    # diversity. Only drop a case whose preflight label can never be refined.
    _SKIP_PREFLIGHT = {"REVERT"}
    remaining = [c for c in remaining if c.get("preflight_action", "?") not in _SKIP_PREFLIGHT]

    # Stratified selection: interleave by preflight_action so a run gets a
    # BALANCED mix of failure types instead of the CODE_FIX-heavy front of the
    # file. After dropping REVERT the dataset is ~51% CODE_FIX / 31% PIN_VERSION
    # / 18% PR_MERGED; taking the first N over-samples CODE_FIX (RPA's default
    # class) and hides the dependency cases where APA can actually beat the baseline.
    from collections import defaultdict, deque
    buckets = defaultdict(deque)
    for c in remaining:
        buckets[c.get("preflight_action", "?")].append(c)
    todo = []
    order = sorted(buckets, key=lambda k: -len(buckets[k]))  # round-robin, largest first
    while len(todo) < args.n and any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                todo.append(buckets[k].popleft())
                if len(todo) >= args.n:
                    break

    print(f"Provider: {os.environ['LLM_PROVIDER']} | cheap={os.environ['CI_AGENT_MODEL']} | "
          f"reason={os.environ['CI_AGENT_CLASSIFY_MODEL']}")
    print(f"Dataset: {len(cases)} | already done: {len(done)} | this run: {len(todo)} | budget: ${args.budget}")

    processed = 0
    for i, c in enumerate(todo, 1):
        rid = c.get("intake", {}).get("run_id", "")
        try:
            check_budget(args.budget)   # clean abort point BEFORE spending on this case
        except BudgetExceeded as e:
            print(f"\n[budget] {e}. Stopping cleanly; {processed} cases done this run. Re-run to continue.")
            break
        print(f"\n[{i}/{len(todo)}] {c.get('intake',{}).get('repo','?')}  (spend ${get_session_cost():.4f})")
        try:
            rec = run_case(c, client)
            append_result(rec)
            processed += 1
            rv = rec["rpa"]["judge"]["verdict"]; av = rec["apa"]["judge"]["verdict"]
            print(f"    rpa={rv} apa={av} gt={rec['ground_truth']['action']}")
        except BudgetExceeded as e:
            print(f"\n[budget] hit mid-case: {e}. Stopping; case not saved. Re-run to continue.")
            break
        except Exception as e:
            print(f"    ERROR on {rid}: {e}")
            traceback.print_exc(limit=2)

    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
