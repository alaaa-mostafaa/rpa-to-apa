# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Dump the same full context the primary full-context judge used, so the secondary LLM
# judge can re-judge the identical evidence independently. Capped to keep it readable.
import json, os, sys, gzip
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import _focus_log_on_error, _failed_step_summary, _fetch_commit_diff

ds = {}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f,"rt",encoding="utf-8"):
            if l.strip(): r=json.loads(l); ds[r["intake"]["run_id"]]=r
rows={json.loads(l)["run_id"]:json.loads(l) for l in open("data/eval_big100.jsonl",encoding="utf-8") if l.strip()}
fj=json.load(open("data/fine_judge_results.json"))

def ctx(rid):
    case=ds.get(rid,{}); ext=case.get("extraction",{})
    excs=[e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e,dict) and e.get("text")]
    focused=_focus_log_on_error(excs,1700).strip()
    errs=ext.get("sample_error_lines") or rows.get(rid,{}).get("error_lines") or []
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
    return "\n".join(parts)[:2400]

out=[]
for i,x in enumerate(fj):
    out.append(dict(i=i,repo=x["repo"],pred=x["pred"],gpt_full=None,ctx=ctx(x["run_id"])))
    if (i+1)%30==0: print(f"  {i+1}/{len(fj)}",flush=True)
json.dump(out,open("data/_fullctx_for_judge2.json","w"),indent=0)
print("wrote",len(out),"chars",sum(len(o['ctx']) for o in out))
