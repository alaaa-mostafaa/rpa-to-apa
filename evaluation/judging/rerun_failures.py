# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Step 3: re-run the AGENT (with the anti-giving-up ACTION_PROMPT) ONLY on cases that were
# judged UNHELPFUL, then re-judge those with the fixed helpfulness judge. Merge back into the
# full result so we get an updated full-corpus number. Honest: only the failures get a second
# attempt with the improved prompt; already-helpful cases are untouched.
import json, os, sys, collections, gzip
sys.path.insert(0, ".")
os.environ["CI_AGENT_SKIP_ACTION"] = "0"
os.environ["CI_AGENT_CLASSIFY_MODEL"] = "deepseek-reasoner"
os.environ["CI_AGENT_MAX_STEPS"] = "5"
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import (build_agent_graph, _initial_tools_for_event, _fetch_commit_diff,
                           _summarize_patch_files, _fetch_run_history, _failed_step_summary,
                           _focus_log_on_error)
from src.apa.bayesian_tracker import BeliefState
from src.apa.llm_config import make_client
from dataclasses import asdict

ds = {}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            if l.strip():
                r = json.loads(l); ds[r["intake"]["run_id"]] = r
rows = {r["run_id"]: r for r in (json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip())}

# start from the re-judged full result
base = json.load(open("data/remediation_rejudged.json"))
key = "cause_v2" if "cause_v2" in base[0] else "cause"
failures = [o for o in base if o.get(key) == "UNHELPFUL" and o["run_id"] in ds]
print(f"re-running agent on {len(failures)} UNHELPFUL cases (anti-giving-up prompt)")

def run_agent(c):
    event = R._event_from_case(c); kwargs = asdict(event)
    ext = c.get("extraction", {}); excs = ext.get("log_excerpts", [])
    log_texts = [e.get("text","") for e in excs if isinstance(e, dict) and "text" in e]
    error_lines = ext.get("sample_error_lines", [])
    cf, cds = [], {}
    if event.repo and event.commit_sha:
        cf = _fetch_commit_diff(event.repo, event.commit_sha).get("files", []); cds = _summarize_patch_files(cf)
    rh = {}
    if event.repo and event.branch and event.run_number:
        try: rh = _fetch_run_history(event.repo, event.branch, event.workflow or "", event.run_number)
        except Exception: pass
    p = BeliefState()
    st = {"run_event": kwargs, "raw_run": c, "beliefs": dict(p.probabilities), "belief_history": list(p.history),
          "confidence": p.confidence(), "entropy": p.entropy(), "tools_available": _initial_tools_for_event(c, event),
          "tools_called": [], "investigation_log": [], "current_step": 0, "done": False, "error_lines": error_lines,
          "mentioned_files": ext.get("mentioned_files", []), "log_excerpt_texts": log_texts, "changed_files": cf,
          "commit_diff": cds, "failed_step_context": _failed_step_summary(event), "dependency_changes": {},
          "run_history": rh, "similar_failures": [], "workflow_contents": [], "runner_environment": {}, "pr_context": {},
          "semantic_diff_links": [], "preprocessing_summary": {"top_category": "CODE_REGRESSION"}, "classification": {},
          "api_key": "", "model": os.environ["CI_AGENT_MODEL"], "_next_tool": ""}
    return build_agent_graph().invoke(st).get("classification", {})

oai = make_client(provider="openai")
_here = os.path.dirname(os.path.abspath(__file__))
JUDGE = open(os.path.join(_here, "rejudge_remediation.py")).read().split('JUDGE = """',1)[1].split('"""',1)[0]

def judge(repo, errs, rec, gt):
    j = oai.chat.completions.create(model="gpt-4o", temperature=0.0, max_tokens=120,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":JUDGE.format(repo=repo, commit="", errors=errs[:1200], rec=rec[:700], gt=gt[:1100])}])
    try: return json.loads(j.choices[0].message.content)
    except Exception: return {"verdict":"PARSE_ERROR","reason":""}

def evidence(rid, fb):
    ext = ds.get(rid, {}).get("extraction", {})
    excs = [e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e, dict) and e.get("text")]
    return _focus_log_on_error(excs, 1400).strip() or fb or "(generic exit code)"

improved = 0
by_id = {o["run_id"]: o for o in base}
for i, o in enumerate(failures, 1):
    rid = o["run_id"]; r = rows.get(rid, {})
    cls = run_agent(ds[rid]); rec = cls.get("recommended_action","") or "(none)"
    gt = str((r.get("ground_truth") or {}).get("reasoning",""))
    errs = evidence(rid, "; ".join((r.get("error_lines") or [])[:5]))
    v = judge(o["repo"], errs, rec, gt).get("verdict")
    if v in ("HELPFUL","PARTIAL"):
        improved += 1
    by_id[rid]["cause_v2"] = v; by_id[rid]["recommendation"] = rec
    print(f"[{i}/{len(failures)}] {o['repo'][:26]:26} UNHELPFUL -> {v}")

final = list(by_id.values())
json.dump(final, open("data/remediation_final.json","w"), indent=2)
c = collections.Counter(o.get("cause_v2", o.get("cause")) for o in final)
hp = c.get("HELPFUL",0)+c.get("PARTIAL",0)
print(f"\n=== FINAL after agent re-run on failures ===")
for k,n in c.most_common(): print(f"  {k:10} {n} ({n/len(final):.0%})")
print(f"  HELPFUL+PARTIAL: {hp}/{len(final)} ({hp/len(final):.0%})")
print(f"  (rescued {improved} of {len(failures)} previously-UNHELPFUL cases)")
