# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Track 2 (full-context evidence judge) for L3 categories.
# L3 category == APA category except on the 40 retrieval rescues (-> prior_top).
# Reuse APA's stored verdict where the category is unchanged; only re-judge the changed ones.
import json, os, sys, collections, gzip
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import _focus_log_on_error, _failed_step_summary, _fetch_commit_diff
from src.apa.llm_config import make_client

ds = {}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f,"rt",encoding="utf-8"):
            if l.strip(): r=json.loads(l); ds[r["intake"]["run_id"]]=r
rows={json.loads(l)["run_id"]:json.loads(l) for l in open("data/eval_big100.jsonl",encoding="utf-8") if l.strip()}
labels=json.load(open("data/evidence_labels_v2.json"))
cats=json.load(open("data/_l3_cats.json"))   # [run_id, apa_cat, l3_cat, apa_defensible]

def rich_context(rid):
    case=ds.get(rid,{}); ext=case.get("extraction",{})
    excs=[e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e,dict) and e.get("text")]
    focused=_focus_log_on_error(excs,6000).strip()
    errs=ext.get("sample_error_lines") or []
    parts=[]
    try:
        event=R._event_from_case(case); step=_failed_step_summary(event)
        if step: parts.append(f"FAILED STEP: {step}")
        if event.repo and event.commit_sha:
            cf=_fetch_commit_diff(event.repo,event.commit_sha).get("files",[])
            changed=[f.get("filename") for f in cf if f.get("filename")][:20]
            if changed: parts.append("FILES CHANGED IN THE COMMIT THAT TRIGGERED THE FAILURE: "+", ".join(changed))
    except Exception: pass
    if errs: parts.append("ERROR LINES:\n"+"\n".join(str(e) for e in errs[:12]))
    parts.append("FAILING LOG (error-focused):\n"+(focused or "(no error region located)"))
    return "\n\n".join(parts)[:7000]

oai=make_client(provider="openai")
JUDGE="""You are classifying why a CI/CD build failed, into one of these fine categories:
CODE_REGRESSION, DEPENDENCY_CONFLICT, CONFIG_ERROR, QUALITY_VIOLATION, TEST_FLAKINESS,
INFRA_INCOMPATIBILITY, ENV_FLAKINESS, CASCADE_FAILURE, TOOLING_ARTIFACT.
A CI failure often legitimately fits MORE THAN ONE of these. You are given the full context the tool
had (the failing log, the error lines, the failed step, and the files the triggering commit changed).
Decide whether the TOOL'S category is a DEFENSIBLE reading of this evidence. Credit it if so.
Mark it wrong ONLY if the evidence clearly contradicts it.
FULL CONTEXT:
{ev}
TOOL'S CATEGORY: {pred}
Respond ONLY JSON: {{"defensible": true/false, "best": "<single most-likely category>", "reason": "one sentence"}}"""

out=[]; rejudged=0
for rid, apa_cat, l3_cat, apa_def in cats:
    if l3_cat==apa_cat:
        deff=apa_def                       # unchanged -> reuse APA verdict
    else:
        ev=rich_context(rid)
        j=oai.chat.completions.create(model="gpt-4o",temperature=0.0,max_tokens=120,
            response_format={"type":"json_object"},
            messages=[{"role":"user","content":JUDGE.format(ev=ev,pred=l3_cat)}])
        try: deff=json.loads(j.choices[0].message.content).get("defensible")
        except Exception: deff=None
        rejudged+=1
        print(f"  rejudged {rid[:40]:40} {apa_cat} -> {l3_cat}  {'DEF' if deff else 'no'}",flush=True)
    inf=labels.get(rid,{}).get("primary") not in (None,"UNINFORMATIVE")
    out.append(dict(run_id=rid,l3_cat=l3_cat,defensible=deff,informative=inf))

json.dump(out,open("data/fine_judge_l3_300.json","w"),indent=1)
n=len(out); nd=sum(1 for o in out if o["defensible"])
print(f"\n=== L3 FINE 9-CATEGORY, full-context judge (re-judged {rejudged}/{n}) ===")
print(f"  L3 defensible (all): {nd}/{n} = {nd/n:.0%}")
