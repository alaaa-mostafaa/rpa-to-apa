# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Dump full context for ALL scorable cases so the secondary LLM judge can re-judge the same
# evidence the primary judge saw. Capped ~2200 chars/case to stay readable. Deterministic scor order.
import json, os, sys, gzip
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import _focus_log_on_error, _failed_step_summary, _fetch_commit_diff
from evals.coarse_eval import substantive_fix_buckets, _load_expert_revisions

rev=_load_expert_revisions()
ds={}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f,"rt",encoding="utf-8"):
            if l.strip(): r=json.loads(l); ds[r["intake"]["run_id"]]=r
rows=[json.loads(l) for l in open("data/eval_big100.jsonl",encoding="utf-8") if l.strip()]
scor=[]
for r in rows:
    rid=r["run_id"]
    if rev.get(rid,{}).get("action")=="NOT_SCORABLE": continue
    pr,_=substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if pr is None or rid not in ds: continue
    pred=((r.get("apa",{}) or {}).get("prediction",{}) or {}).get("category")
    if not pred: continue
    scor.append((rid,r["repo"],pred))

def ctx(rid):
    case=ds.get(rid,{}); ext=case.get("extraction",{})
    excs=[e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e,dict) and e.get("text")]
    focused=_focus_log_on_error(excs,1500).strip()
    errs=ext.get("sample_error_lines") or []
    parts=[]
    try:
        event=R._event_from_case(case); step=_failed_step_summary(event)
        if step: parts.append(f"STEP: {step}")
        if event.repo and event.commit_sha:
            cf=_fetch_commit_diff(event.repo,event.commit_sha).get("files",[])
            changed=[f.get("filename") for f in cf if f.get("filename")][:15]
            if changed: parts.append("COMMIT CHANGED FILES: "+", ".join(changed))
    except Exception: pass
    if errs: parts.append("ERRORS: "+" | ".join(str(e) for e in errs[:8]))
    parts.append("LOG:\n"+(focused or "(no error region)"))
    return "\n".join(parts)[:2200]

out=[]
for i,(rid,repo,pred) in enumerate(scor):
    out.append(dict(i=i,repo=repo,pred=pred,ctx=ctx(rid)))
    if (i+1)%40==0: print(f"  {i+1}/{len(scor)}",flush=True)
json.dump(out,open("data/_fullctx_300_for_judge2.json","w"),indent=0)
print("wrote",len(out),"total chars",sum(len(o['ctx']) for o in out))
