# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Fair re-judge of the fine 9-category predictions, but giving the judge the FULL case
# context the classifier actually had (large error-focused log + error lines + failed step
# + files mentioned + the commit's changed files), instead of a 1400-char keyhole snippet.
# Same lenient multi-category rubric. GPT-4o is judge 1. Predictions are the STORED ones
# (data/fine_judge_results.json) -- we are re-judging, not re-classifying.
import json, os, sys, collections, gzip
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import _focus_log_on_error, _failed_step_summary, _fetch_commit_diff, _summarize_patch_files
from src.apa.llm_config import make_client

ds = {}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f,"rt",encoding="utf-8"):
            if l.strip(): r=json.loads(l); ds[r["intake"]["run_id"]]=r
rows={json.loads(l)["run_id"]:json.loads(l) for l in open("data/eval_big100.jsonl",encoding="utf-8") if l.strip()}
fj=json.load(open("data/fine_judge_results.json"))
print(f"re-judging {len(fj)} cases with FULL context", flush=True)

def rich_context(rid):
    case=ds.get(rid,{}); ext=case.get("extraction",{})
    excs=[e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e,dict) and e.get("text")]
    focused=_focus_log_on_error(excs,6000).strip()
    errs=ext.get("sample_error_lines") or rows.get(rid,{}).get("error_lines") or []
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

A CI failure often legitimately fits MORE THAN ONE of these. You are given the full context the tool had.
Decide whether the TOOL'S category is a DEFENSIBLE reading of the evidence, i.e. a reasonable engineer
could assign it given this context. Credit it if so. Mark it wrong ONLY if the evidence clearly contradicts it.

FULL CONTEXT:
{ev}

TOOL'S CATEGORY: {pred}

Respond ONLY JSON: {{"defensible": true/false, "best": "<single most-likely category>", "reason": "one sentence"}}"""

out=[]
for i,x in enumerate(fj,1):
    rid=x["run_id"]; pred=x["pred"]
    ev=rich_context(rid)
    j=oai.chat.completions.create(model="gpt-4o",temperature=0.0,max_tokens=120,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":JUDGE.format(ev=ev,pred=pred)}])
    try: v=json.loads(j.choices[0].message.content)
    except Exception: v={"defensible":None,"best":None,"reason":""}
    out.append(dict(run_id=rid,repo=x["repo"],pred=pred,
                    defensible=v.get("defensible"),best=v.get("best"),reason=v.get("reason"),
                    keyhole_defensible=x["defensible"]))
    mark="DEF" if v.get("defensible") else "no "
    print(f"[{i}/{len(fj)}] {mark} {x['repo'][:34]:34} {pred}",flush=True)

json.dump(out,open("data/fine_judge_fullctx.json","w"),indent=1)
n=len(out); nd=sum(1 for o in out if o["defensible"]); kh=sum(1 for o in out if o["keyhole_defensible"])
flipped=sum(1 for o in out if o["defensible"] and not o["keyhole_defensible"])
lost=sum(1 for o in out if (not o["defensible"]) and o["keyhole_defensible"])
print(f"\n=== FINE 9-CATEGORY, GPT-4o judge WITH FULL CONTEXT, n={n} ===")
print(f"  keyhole (1400-char) defensible: {kh}/{n} = {kh/n:.0%}")
print(f"  full-context     defensible:    {nd}/{n} = {nd/n:.0%}")
print(f"  rescued by full context (no->yes): {flipped}   lost (yes->no): {lost}")
per=collections.defaultdict(lambda:[0,0])
for o in out: per[o["pred"]][1]+=1; per[o["pred"]][0]+=(1 if o["defensible"] else 0)
for c in sorted(per,key=lambda c:-per[c][1]):
    print(f"    {c:22} n={per[c][1]:3}  defensible={per[c][0]/per[c][1]:.0%}")
