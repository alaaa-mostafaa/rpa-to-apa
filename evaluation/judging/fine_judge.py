# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Fine-grained (9-category) evaluation with a FAIR multi-category judge.
# The classifier already predicts one of 9 fine categories. This re-judges the stored
# APA prediction: given the failing-log evidence, is APA's fine category a DEFENSIBLE
# diagnosis? CI failures legitimately fit multiple fine categories (a failing test may be
# CODE_REGRESSION, QUALITY_VIOLATION, or TEST_FLAKINESS), so the judge credits any fine
# category genuinely supported by the evidence. It does NOT rubber-stamp: a category
# contradicted by the evidence is marked wrong. GPT-4o is judge 1; a secondary LLM judge is judge 2.
import json, os, sys, collections, gzip
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
from src.apa.llm_config import make_client
from src.apa.agent import _focus_log_on_error

CATS = ["CODE_REGRESSION","DEPENDENCY_CONFLICT","CONFIG_ERROR","QUALITY_VIOLATION",
        "TEST_FLAKINESS","INFRA_INCOMPATIBILITY","ENV_FLAKINESS","CASCADE_FAILURE","TOOLING_ARTIFACT"]

ds = {}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz",
          "data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f,"rt",encoding="utf-8"):
            if l.strip():
                r=json.loads(l); ds[r["intake"]["run_id"]]=r
rows=[json.loads(l) for l in open("data/eval_big100.jsonl",encoding="utf-8") if l.strip()]
labels=json.load(open("data/evidence_labels_v2.json"))

def evidence(rid, fb):
    ext=ds.get(rid,{}).get("extraction",{})
    excs=[e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e,dict) and e.get("text")]
    return _focus_log_on_error(excs,1400).strip() or fb or "(generic exit code)"

# only score cases the evidence judge found informative (a real diagnosable error)
scor=[r for r in rows if (labels.get(r["run_id"],{}).get("primary") not in (None,"UNINFORMATIVE"))]
print(f"fine-grained judging on {len(scor)} informative cases", flush=True)

oai=make_client(provider="openai")
JUDGE="""You are classifying why a CI/CD build failed, into one of these fine categories:
CODE_REGRESSION (a code change broke a test/build), DEPENDENCY_CONFLICT (a dependency/version/lockfile issue),
CONFIG_ERROR (workflow/CI/YAML/secret/config), QUALITY_VIOLATION (lint/format/style/coverage gate),
TEST_FLAKINESS (intermittent test), INFRA_INCOMPATIBILITY (runner OS/toolchain/arch), ENV_FLAKINESS (network/registry/transient infra),
CASCADE_FAILURE (failed only because an upstream job failed), TOOLING_ARTIFACT (the CI tool itself, not the project).

A CI failure often fits MORE THAN ONE of these. Decide whether the TOOL'S category is a DEFENSIBLE
reading of the evidence, i.e. a reasonable engineer could assign it given the log. Credit it if so.
Mark it wrong only if the evidence clearly contradicts it.

FAILING-LOG EVIDENCE:
{ev}

TOOL'S CATEGORY: {pred}

Respond ONLY JSON: {{"defensible": true/false, "best": "<the single most-likely category>", "reason": "one sentence"}}"""

out=[]
for i,r in enumerate(scor,1):
    rid=r["run_id"]; pred=(r.get("apa",{}).get("prediction") or {}).get("category")
    if not pred: continue
    ev=evidence(rid,"; ".join((r.get("error_lines") or [])[:5]))
    j=oai.chat.completions.create(model="gpt-4o",temperature=0.0,max_tokens=120,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":JUDGE.format(ev=ev[:1400],pred=pred)}])
    try: v=json.loads(j.choices[0].message.content)
    except Exception: v={"defensible":None,"best":None,"reason":""}
    out.append(dict(run_id=rid,repo=r["repo"],pred=pred,
                    defensible=v.get("defensible"),best=v.get("best"),reason=v.get("reason"),
                    evidence_label=labels.get(rid,{}).get("primary")))
    if i%25==0: print(f"  {i}/{len(scor)}",flush=True)

json.dump(out,open("data/fine_judge_results.json","w"),indent=1)
n=len(out); ndef=sum(1 for o in out if o["defensible"]); exact=sum(1 for o in out if o["pred"]==o["best"])
print(f"\n=== FINE 9-CATEGORY (GPT-4o fair multi-category judge), n={n} ===")
print(f"  APA category DEFENSIBLE: {ndef}/{n} = {ndef/n:.0%}")
print(f"  APA == judge's single best: {exact}/{n} = {exact/n:.0%}")
print("  per predicted category (n, defensible%):")
per=collections.defaultdict(lambda:[0,0])
for o in out: per[o["pred"]][1]+=1; per[o["pred"]][0]+= (1 if o["defensible"] else 0)
for c in sorted(per,key=lambda c:-per[c][1]):
    print(f"    {c:22} n={per[c][1]:3}  defensible={per[c][0]/per[c][1]:.0%}")
