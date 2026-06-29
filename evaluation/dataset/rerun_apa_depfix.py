#!/usr/bin/env python3
"""
rerun_apa_depfix.py
===================
Re-run ONLY the APA agent on the already-selected balanced cases, to measure the
dependency-detection fix in isolation. Reuses the stored ground-truth action and
RPA prediction (so no GT re-scrape, no judge calls) and only re-invokes the agent
+ commit-diff fetch. Writes data/eval_balanced_depfix.jsonl, then coarse-scores.
"""
from __future__ import annotations

# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
import json, os, sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_eval_500 as R  # importing sets provider/model env vars
from src.apa.agent import (
    build_agent_graph, _initial_tools_for_event, _fetch_commit_diff,
    _summarize_patch_files, _fetch_run_history, _failed_step_summary,
)
from src.apa.bayesian_tracker import BeliefState

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/eval_balanced_depfix.jsonl"
IN = sys.argv[2] if len(sys.argv) > 2 else "data/eval_balanced_results.jsonl"


def apa_classify(c: dict, rpa_category: str) -> dict:
    event = R._event_from_case(c)
    kwargs = asdict(event)
    extraction = c.get("extraction", {})
    excerpts = extraction.get("log_excerpts", [])
    log_texts = [e.get("text", "") for e in excerpts if isinstance(e, dict) and "text" in e]
    error_lines = extraction.get("sample_error_lines", [])

    changed_files, commit_diff_summary = [], {}
    if event.repo and event.commit_sha:
        changed_files = _fetch_commit_diff(event.repo, event.commit_sha).get("files", [])
        commit_diff_summary = _summarize_patch_files(changed_files)
    run_history = {}
    if event.repo and event.branch and event.run_number:
        try:
            run_history = _fetch_run_history(event.repo, event.branch, event.workflow or "", event.run_number)
        except Exception:
            pass

    prior = BeliefState()
    state = {
        "run_event": kwargs, "raw_run": c,
        "beliefs": dict(prior.probabilities), "belief_history": list(prior.history),
        "confidence": prior.confidence(), "entropy": prior.entropy(),
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
        "preprocessing_summary": {"top_category": rpa_category},
        "classification": {}, "api_key": "",
        "model": os.environ["CI_AGENT_MODEL"], "_next_tool": "",
    }
    return build_agent_graph().invoke(state).get("classification", {})


def main():
    cases = {c.get("intake", {}).get("run_id", ""): c for c in R.load_cases()}
    saved = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    print(f"re-running APA on {len(saved)} balanced cases (dep-fix). spend cap not enforced here.")
    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(saved, 1):
            rid = r["run_id"]; c = cases.get(rid)
            if not c:
                print(f"  [{i}] MISSING dataset case {rid}"); continue
            rpa_cat = (r.get("rpa", {}).get("prediction") or {}).get("category", "CODE_REGRESSION")
            try:
                apa = apa_classify(c, rpa_cat)
            except Exception as e:
                print(f"  [{i}] ERROR {rid[:40]}: {e}"); apa = {}
            rec = dict(r)
            rec["apa"] = {"prediction": apa, "judge": r.get("apa", {}).get("judge", {})}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(saved)}] {rid[:46]:46} -> {apa.get('category')}")
    print("\n========== COARSE (dep-fix re-run) ==========")
    from evals.coarse_eval import score, load_results
    score(load_results([OUT]))


if __name__ == "__main__":
    raise SystemExit(main())
