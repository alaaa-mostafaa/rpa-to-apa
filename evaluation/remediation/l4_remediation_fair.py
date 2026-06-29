# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# FAIR remediation eval: re-run the APA agent with the recommended-action step
# ENABLED (full evidence), capturing its REAL 2-3 sentence fix, then have GPT-4o
# judge whether that fix would have helped vs the developer's actual fix.
import json, os, sys, collections, gzip
sys.path.insert(0, ".")
os.environ["CI_AGENT_SKIP_ACTION"] = "0"            # enable the real fix step
os.environ["CI_AGENT_CLASSIFY_MODEL"] = "deepseek-reasoner"
os.environ["CI_AGENT_MAX_STEPS"] = "5"
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R  # importing sets provider/model env (setdefault won't override the above)
from src.apa.agent import (build_agent_graph, _initial_tools_for_event, _fetch_commit_diff,
                           _summarize_patch_files, _fetch_run_history, _failed_step_summary)
from src.apa.bayesian_tracker import BeliefState
from src.apa.llm_config import make_client
from evals.coarse_eval import substantive_fix_buckets, _load_expert_revisions
from dataclasses import asdict

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
rev = _load_expert_revisions()

# dataset cases by run_id (for full evidence)
ds = {}
for f in ("data/dataset_remote_250.jsonl.gz", "data/dataset_remote_120.jsonl.gz", "data/dataset_remote_next.jsonl.gz"):
    try:
        for l in gzip.open(f, "rt", encoding="utf-8"):
            if l.strip():
                r = json.loads(l); ds[r["intake"]["run_id"]] = r
    except Exception:
        pass

rows = [json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip()]
scor = []
for r in rows:
    if rev.get(r["run_id"], {}).get("action") == "NOT_SCORABLE":
        continue
    pr, _ = substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if pr is None or r["run_id"] not in ds:
        continue
    scor.append(r)
scor = scor[:N]
print(f"fair remediation eval on {len(scor)} cases (real APA recommendation + GPT-4o judge)")

def run_agent_with_action(c):
    event = R._event_from_case(c); kwargs = asdict(event)
    ext = c.get("extraction", {}); excs = ext.get("log_excerpts", [])
    log_texts = [e.get("text", "") for e in excs if isinstance(e, dict) and "text" in e]
    error_lines = ext.get("sample_error_lines", [])
    cf, cds = [], {}
    if event.repo and event.commit_sha:
        cf = _fetch_commit_diff(event.repo, event.commit_sha).get("files", []); cds = _summarize_patch_files(cf)
    rh = {}
    if event.repo and event.branch and event.run_number:
        try: rh = _fetch_run_history(event.repo, event.branch, event.workflow or "", event.run_number)
        except Exception: pass
    prior = BeliefState()
    st = {"run_event": kwargs, "raw_run": c, "beliefs": dict(prior.probabilities),
          "belief_history": list(prior.history), "confidence": prior.confidence(), "entropy": prior.entropy(),
          "tools_available": _initial_tools_for_event(c, event), "tools_called": [], "investigation_log": [],
          "current_step": 0, "done": False, "error_lines": error_lines,
          "mentioned_files": ext.get("mentioned_files", []), "log_excerpt_texts": log_texts,
          "changed_files": cf, "commit_diff": cds, "failed_step_context": _failed_step_summary(event),
          "dependency_changes": {}, "run_history": rh, "similar_failures": [], "workflow_contents": [],
          "runner_environment": {}, "pr_context": {}, "semantic_diff_links": [],
          "preprocessing_summary": {"top_category": "CODE_REGRESSION"}, "classification": {}, "api_key": "",
          "model": os.environ["CI_AGENT_MODEL"], "_next_tool": ""}
    return build_agent_graph().invoke(st).get("classification", {})

oai = make_client(provider="openai")
JUDGE = """You are a senior engineer reviewing an automated CI-triage tool. A run failed; the tool produced a recommended fix. You are told what the developer ACTUALLY did. Judge whether the tool's recommendation would have led the developer to substantially the correct fix.

FAILURE
  Repo: {repo}
  Commit: {commit}
  Error lines: {errors}

TOOL'S RECOMMENDED FIX:
  {rec}

WHAT THE DEVELOPER ACTUALLY DID:
  {gt}

Respond ONLY as JSON: {{"verdict": "HELPFUL|PARTIAL|UNHELPFUL", "reason": "one sentence"}}"""

out = []
for i, r in enumerate(scor, 1):
    cls = run_agent_with_action(ds[r["run_id"]])
    rec = cls.get("recommended_action", "") or "(none)"
    errs = "; ".join((r.get("error_lines") or [])[:5]) or "(generic exit code)"
    j = oai.chat.completions.create(model="gpt-4o", temperature=0.0, max_tokens=120,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE.format(repo=r["repo"], commit=r.get("commit", "")[:100],
            errors=errs[:400], rec=rec[:500], gt=str(r["ground_truth"].get("reasoning", ""))[:600])}])
    try: v = json.loads(j.choices[0].message.content)
    except Exception: v = {"verdict": "PARSE_ERROR", "reason": ""}
    out.append(dict(run_id=r["run_id"], repo=r["repo"], category=cls.get("category"),
                    recommendation=rec, verdict=v.get("verdict"), reason=v.get("reason")))
    print(f"[{i}/{len(scor)}] {r['repo'][:28]:28} {cls.get('category','?'):18} -> {v.get('verdict')}")

json.dump(out, open("data/remediation_eval_fair.json", "w"), indent=2)
dist = collections.Counter(o["verdict"] for o in out)
print("\n=== Remediation quality (real APA fix, GPT-4o judge) ===")
for k, n in dist.most_common():
    print(f"  {k:10} {n} ({n/len(out):.0%})")
helpful = dist.get("HELPFUL", 0) + dist.get("PARTIAL", 0)
print(f"  HELPFUL+PARTIAL: {helpful}/{len(out)} ({helpful/len(out):.0%})")
print("\n--- 3 samples ---")
for o in out[:3]:
    print(f"  {o['repo'][:26]} [{o['verdict']}] cat={o['category']}")
    print(f"    FIX: {o['recommendation'][:160]}")
    print(f"    JUDGE: {o['reason']}")
