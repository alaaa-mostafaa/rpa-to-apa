# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Re-judge SAVED remediation recommendations with the fixed helpfulness judge
# (source-vs-test location understanding + full developer changed-file list).
# Cheap: only GPT-4o judge calls, NO agent re-run.
import json, sys, collections
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
from src.apa.llm_config import make_client
from src.apa.agent import _focus_log_on_error
import gzip, os

ds = {}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            if l.strip():
                r = json.loads(l); ds[r["intake"]["run_id"]] = r
rows = {r["run_id"]: r for r in (json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip())}

JUDGE = """You are a senior engineer judging whether an automated CI-triage tool's recommended fix would resolve a failed build. The developer's actual change is ONE reference (not the only valid fix).

JUDGING RULES:
- HELPFUL: a plausible, concrete fix that addresses the actual error. Credit a DIFFERENT-but-valid fix; reverting a breaking commit counts.
- CRITICAL source-vs-test: a TEST failure is fixed by changing the SOURCE code under test, not the test file. If the tool names a SOURCE file the developer ALSO changed, it is HELPFUL even though the error appeared in a test file. Never penalize a fix for being "in a different file than where the error appears".
- MATCH ON DEVELOPER'S CHANGED FILES: if the file the tool names is in the developer's changed-file list, that is strong evidence of HELPFUL.
- PARTIAL: right subsystem but incomplete.
- UNHELPFUL: wrong area (not in the developer's changed files AND contradicted by the error), gives up ("re-run/flaky") on a real defect, or too vague.
When torn, pick the MORE generous grade.

FAILURE  Repo: {repo}  Commit: {commit}
Error / log evidence: {errors}
TOOL'S RECOMMENDED FIX: {rec}
DEVELOPER'S ACTUAL CHANGE (reference; note the changed files): {gt}

Respond ONLY as JSON: {{"verdict":"HELPFUL|PARTIAL|UNHELPFUL","reason":"one sentence"}}"""

def evidence(run_id, fallback):
    c = ds.get(run_id, {}); ext = c.get("extraction", {}) if c else {}
    excs = [e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e, dict) and e.get("text")]
    t = _focus_log_on_error(excs, max_chars=1400)
    return t.strip() or fallback or "(generic exit code)"

oai = make_client(provider="openai")
src = sys.argv[1] if len(sys.argv) > 1 else "data/remediation_eval_dual.json"
data = json.load(open(src))
out = []
for i, o in enumerate(data, 1):
    rid = o["run_id"]; r = rows.get(rid, {})
    gt = str((r.get("ground_truth") or {}).get("reasoning",""))[:1100]
    errs = evidence(rid, "; ".join((r.get("error_lines") or [])[:5]))
    j = oai.chat.completions.create(model="gpt-4o", temperature=0.0, max_tokens=120,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":JUDGE.format(repo=o["repo"], commit="",
            errors=errs[:1200], rec=(o.get("recommendation") or "")[:700], gt=gt)}])
    try: v = json.loads(j.choices[0].message.content)
    except Exception: v = {"verdict":"PARSE_ERROR","reason":""}
    o2 = dict(o); o2["cause_v2"] = v.get("verdict"); o2["cause_v2_reason"] = v.get("reason")
    out.append(o2)
    print(f"[{i}/{len(data)}] {o['repo'][:26]:26} {o.get('cause')} -> {v.get('verdict')}")

json.dump(out, open("data/remediation_rejudged.json","w"), indent=2)
c = collections.Counter(o["cause_v2"] for o in out)
hp = c.get("HELPFUL",0)+c.get("PARTIAL",0)
print("\n=== RE-JUDGED (fixed judge) ===")
for k,n in c.most_common(): print(f"  {k:10} {n} ({n/len(out):.0%})")
print(f"  HELPFUL+PARTIAL: {hp}/{len(out)} ({hp/len(out):.0%})")
old = collections.Counter(o.get("cause") for o in out); oldhp = old.get("HELPFUL",0)+old.get("PARTIAL",0)
print(f"  (was: {oldhp}/{len(out)} = {oldhp/len(out):.0%})")
