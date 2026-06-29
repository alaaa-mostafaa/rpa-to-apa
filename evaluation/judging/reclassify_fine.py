# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Clean re-run: re-run the agent classifier (fresh) on the informative cases, capturing the
# fine 9-category prediction, then judge it with the fair multi-category GPT-4o judge.
import json, os, sys, collections, gzip
sys.path.insert(0, ".")
os.environ["CI_AGENT_SKIP_ACTION"] = "1"          # classification only, no fix step (cheaper)
os.environ["CI_AGENT_CLASSIFY_MODEL"] = "deepseek-reasoner"
os.environ["CI_AGENT_MAX_STEPS"] = "5"
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import (build_agent_graph, _initial_tools_for_event, _fetch_commit_diff,
                           _summarize_patch_files, _fetch_run_history, _failed_step_summary, _focus_log_on_error)
from src.apa.bayesian_tracker import BeliefState
from src.apa.llm_config import make_client
from dataclasses import asdict

ds={}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f,"rt",encoding="utf-8"):
            if l.strip(): r=json.loads(l); ds[r["intake"]["run_id"]]=r
rows={r["run_id"]:r for r in (json.loads(l) for l in open("data/eval_big100.jsonl",encoding="utf-8") if l.strip())}
labels=json.load(open("data/evidence_labels_v2.json"))
scor=[rid for rid in labels if labels[rid].get("primary") not in (None,"UNINFORMATIVE") and rid in ds and rid in rows]
print(f"re-classifying {len(scor)} informative cases (fresh 9-category run)", flush=True)

def reclassify(c):
    event=R._event_from_case(c); kwargs=asdict(event)
    ext=c.get("extraction",{}); excs=ext.get("log_excerpts",[])
    log_texts=[e.get("text","") for e in excs if isinstance(e,dict) and "text" in e]
    error_lines=ext.get("sample_error_lines",[])
    cf,cds=[],{}
    if event.repo and event.commit_sha:
        cf=_fetch_commit_diff(event.repo,event.commit_sha).get("files",[]); cds=_summarize_patch_files(cf)
    rh={}
    if event.repo and event.branch and event.run_number:
        try: rh=_fetch_run_history(event.repo,event.branch,event.workflow or "",event.run_number)
        except Exception: pass
    p=BeliefState()
    st={"run_event":kwargs,"raw_run":c,"beliefs":dict(p.probabilities),"belief_history":list(p.history),
        "confidence":p.confidence(),"entropy":p.entropy(),"tools_available":_initial_tools_for_event(c,event),
        "tools_called":[],"investigation_log":[],"current_step":0,"done":False,"error_lines":error_lines,
        "mentioned_files":ext.get("mentioned_files",[]),"log_excerpt_texts":log_texts,"changed_files":cf,
        "commit_diff":cds,"failed_step_context":_failed_step_summary(event),"dependency_changes":{},
        "run_history":rh,"similar_failures":[],"workflow_contents":[],"runner_environment":{},"pr_context":{},
        "semantic_diff_links":[],"preprocessing_summary":{"top_category":"CODE_REGRESSION"},"classification":{},
        "api_key":"","model":os.environ["CI_AGENT_MODEL"],"_next_tool":""}
    return build_agent_graph().invoke(st).get("classification",{}).get("category")

oai=make_client(provider="openai")
JUDGE="""You are classifying why a CI/CD build failed, into one of 9 fine categories:
CODE_REGRESSION, DEPENDENCY_CONFLICT, CONFIG_ERROR, QUALITY_VIOLATION, TEST_FLAKINESS,
INFRA_INCOMPATIBILITY, ENV_FLAKINESS, CASCADE_FAILURE, TOOLING_ARTIFACT.
A CI failure often fits MORE THAN ONE. Decide whether the TOOL'S category is a DEFENSIBLE reading
of the evidence (a reasonable engineer could assign it). Credit it if so; mark wrong only if the
evidence clearly contradicts it.
EVIDENCE: {ev}
TOOL'S CATEGORY: {pred}
Respond ONLY JSON: {{"defensible": true/false, "best": "<single most-likely category>"}}"""
def ev_of(rid,fb):
    ext=ds.get(rid,{}).get("extraction",{}); excs=[e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e,dict) and e.get("text")]
    return _focus_log_on_error(excs,1400).strip() or fb or "(generic exit code)"

out=[]
for i,rid in enumerate(scor,1):
    pred=reclassify(ds[rid])
    r=rows[rid]; ev=ev_of(rid,"; ".join((r.get("error_lines") or [])[:5]))
    j=oai.chat.completions.create(model="gpt-4o",temperature=0.0,max_tokens=80,response_format={"type":"json_object"},
        messages=[{"role":"user","content":JUDGE.format(ev=ev[:1400],pred=pred)}])
    try: v=json.loads(j.choices[0].message.content)
    except Exception: v={"defensible":None,"best":None}
    out.append(dict(run_id=rid,repo=r["repo"],pred=pred,defensible=v.get("defensible"),best=v.get("best")))
    print(f"[{i}/{len(scor)}] {r['repo'][:26]:26} {pred} -> {'DEF' if v.get('defensible') else 'no'}",flush=True)

json.dump(out,open("data/fine_reclassified.json","w"),indent=1)
n=len(out); ndef=sum(1 for o in out if o["defensible"]); exact=sum(1 for o in out if o["pred"]==o["best"])
print(f"\n=== FRESH 9-CATEGORY RE-RUN (GPT-4o fair judge), n={n} ===")
print(f"  defensible: {ndef}/{n} = {ndef/n:.0%}")
print(f"  exact-best: {exact}/{n} = {exact/n:.0%}")
per=collections.defaultdict(lambda:[0,0])
for o in out: per[o['pred']][1]+=1; per[o['pred']][0]+=(1 if o['defensible'] else 0)
for c in sorted(per,key=lambda c:-per[c][1]): print(f"    {c:22} n={per[c][1]:3} def={per[c][0]/per[c][1]:.0%}")
