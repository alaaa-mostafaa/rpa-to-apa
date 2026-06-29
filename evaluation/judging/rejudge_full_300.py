# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Fine 9-category fair judge over ALL 300 scorable cases (not just the log-informative 177),
# giving the judge the FULL context the classifier had (large error-focused log + error lines
# + failed step + the commit's changed files). Rationale: the diagnostic signal can legitimately
# come from the diff/changed-files/step even when the log alone is thin, so every scorable case
# is judgeable on full context. GPT-4o, lenient multi-category rubric. Re-judges STORED predictions.
import json, os, sys, collections, gzip
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import _focus_log_on_error, _failed_step_summary, _fetch_commit_diff
from src.apa.llm_config import make_client
from evals.coarse_eval import substantive_fix_buckets, _load_expert_revisions

rev=_load_expert_revisions()
ds={}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f,"rt",encoding="utf-8"):
            if l.strip(): r=json.loads(l); ds[r["intake"]["run_id"]]=r
rows=[json.loads(l) for l in open("data/eval_big100.jsonl",encoding="utf-8") if l.strip()]
labels=json.load(open("data/evidence_labels_v2.json"))
scor=[]
for r in rows:
    rid=r["run_id"]
    if rev.get(rid,{}).get("action")=="NOT_SCORABLE": continue
    pr,_=substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if pr is None or rid not in ds: continue
    pred=((r.get("apa",{}) or {}).get("prediction",{}) or {}).get("category")
    if not pred: continue
    r["_pred"]=pred; scor.append(r)
print(f"judging {len(scor)} scorable cases on full context", flush=True)

def rich_context(rid):
    case=ds.get(rid,{}); ext=case.get("extraction",{})
    excs=[e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e,dict) and e.get("text")]
    focused=_focus_log_on_error(excs,6000).strip()
    errs=ext.get("sample_error_lines") or []
    mentioned=ext.get("mentioned_files") or []
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
    if mentioned: parts.append("FILES MENTIONED IN LOG: "+", ".join(mentioned[:15]))
    parts.append("FAILING LOG (error-focused):\n"+(focused or "(no error region located)"))
    return "\n\n".join(parts)[:7000]

oai=make_client(provider="openai")
JUDGE="""You are classifying why a CI/CD build failed, into one of these fine categories:
CODE_REGRESSION (a code change broke a test/build), DEPENDENCY_CONFLICT (a dependency/version/lockfile issue),
CONFIG_ERROR (workflow/CI/YAML/secret/publish/config), QUALITY_VIOLATION (lint/format/style/coverage/static-analysis gate),
TEST_FLAKINESS (intermittent test), INFRA_INCOMPATIBILITY (runner OS/toolchain/compiler/arch), ENV_FLAKINESS (network/registry/transient infra),
CASCADE_FAILURE (failed only because an upstream job failed), TOOLING_ARTIFACT (the CI tool itself, not the project).

A CI failure often legitimately fits MORE THAN ONE of these. You are given the full context the tool had
(the failing log, the error lines, the failed step, and the files the triggering commit changed). The cause
may be evident from the changed files or step even if the log text alone is thin.
Decide whether the TOOL'S category is a DEFENSIBLE reading of this evidence, i.e. a reasonable engineer
could assign it. Credit it if so. Mark it wrong ONLY if the evidence clearly contradicts it.

FULL CONTEXT:
{ev}

TOOL'S CATEGORY: {pred}

Respond ONLY JSON: {{"defensible": true/false, "best": "<single most-likely category>", "reason": "one sentence"}}"""

# resume from checkpoint: skip cases already judged
CKPT="data/fine_judge_300.json"
out=[]; done=set()
if os.path.exists(CKPT):
    out=json.load(open(CKPT)); done={o["run_id"] for o in out}
    print(f"resuming: {len(done)} already judged, {len(scor)-len(done)} remaining", flush=True)
for i,r in enumerate(scor,1):
    rid=r["run_id"]
    if rid in done: continue
    pred=r["_pred"]; ev=rich_context(rid)
    try:
        j=oai.chat.completions.create(model="gpt-4o",temperature=0.0,max_tokens=120,
            response_format={"type":"json_object"},
            messages=[{"role":"user","content":JUDGE.format(ev=ev,pred=pred)}])
        v=json.loads(j.choices[0].message.content)
    except Exception as e:
        print(f"STOPPED at [{i}/{len(scor)}] {type(e).__name__}: saving {len(out)} and exiting", flush=True)
        json.dump(out,open(CKPT,"w"),indent=1); raise SystemExit(0)
    informative=labels.get(rid,{}).get("primary") not in (None,"UNINFORMATIVE")
    out.append(dict(run_id=rid,repo=r["repo"],pred=pred,defensible=v.get("defensible"),
                    best=v.get("best"),reason=v.get("reason"),informative=informative))
    print(f"[{i}/{len(scor)}] {'DEF' if v.get('defensible') else 'no '} {r['repo'][:32]:32} {pred}",flush=True)
    if len(out)%10==0: json.dump(out,open(CKPT,"w"),indent=1)

json.dump(out,open(CKPT,"w"),indent=1)
n=len(out); nd=sum(1 for o in out if o["defensible"])
inf=[o for o in out if o["informative"]]; uninf=[o for o in out if not o["informative"]]
print(f"\n=== FINE 9-CATEGORY, GPT-4o full-context judge, ALL SCORABLE n={n} ===")
print(f"  defensible (all 300):                 {nd}/{n} = {nd/n:.0%}")
print(f"  defensible (log-informative subset):  {sum(o['defensible'] for o in inf)}/{len(inf)} = {sum(o['defensible'] for o in inf)/max(len(inf),1):.0%}")
print(f"  defensible (log-thin subset):         {sum(o['defensible'] for o in uninf)}/{len(uninf)} = {sum(o['defensible'] for o in uninf)/max(len(uninf),1):.0%}")
per=collections.defaultdict(lambda:[0,0])
for o in out: per[o["pred"]][1]+=1; per[o["pred"]][0]+=(1 if o["defensible"] else 0)
for c in sorted(per,key=lambda c:-per[c][1]):
    print(f"    {c:22} n={per[c][1]:3}  defensible={per[c][0]/per[c][1]:.0%}")
