# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# FAITHFUL L3 re-run, but ground truth = EVIDENCE label (evidence_labels_v2 'primary'),
# not the developer fix. Identical machinery to l3_rerun.py: real ChromaCaseStore,
# compute_prior (trust-weighting + prior blend + 0.72 floor), leave-one-out, same rescue rule.
import json, os, sys, shutil
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
from evals.coarse_eval import pred_bucket, BUCKETS
from src.apa.chroma_case_store import ChromaCaseStore

lab = json.load(open("data/evidence_labels_v2.json"))
rows = [json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip()]
def cat(r, s): return (r.get(s, {}).get("prediction") or {}).get("category")
BUCKET_TO_CAT = {"CODE": "CODE_REGRESSION", "DEPENDENCY": "DEPENDENCY_CONFLICT",
                 "CONFIG": "CONFIG_ERROR", "TRANSIENT": "ENV_FLAKINESS"}

# scorable here = cases with an informative evidence label
scor = []
for r in rows:
    pri = lab.get(r["run_id"], {}).get("primary")
    if pri in (None, "UNINFORMATIVE"): continue
    eb = pred_bucket(pri)                 # evidence label -> coarse family
    if eb is None: continue
    r["_eb"] = eb; scor.append(r)
print(f"L3 evidence re-run on {len(scor)} cases (GT = evidence label)", flush=True)

def apa_verdict(r):
    # APA correctness vs the EVIDENCE family (coarse), mirroring l3_rerun's CORRECT/WRONG
    ab = pred_bucket(cat(r, "apa"))
    return "CORRECT" if ab == r["_eb"] else "WRONG"

CH = "data/chroma_l3_evidence"
shutil.rmtree(CH, ignore_errors=True)
store = ChromaCaseStore(path=CH)
for r in scor:
    store.upsert_case(run_id=r["run_id"], commit_title=r.get("commit", ""),
                      error_lines=r.get("error_lines", []), category=cat(r, "apa"),
                      gt_verdict=apa_verdict(r), repo=r.get("repo", ""))
print(f"store populated: {store.count()} cases", flush=True)

def rank_of(prior, c):
    return sorted(prior, key=prior.get, reverse=True).index(c) + 1 if c in prior else len(prior)

results = []
for i, r in enumerate(scor):
    gt_cat = BUCKET_TO_CAT[r["_eb"]]; av = apa_verdict(r); rid = r["run_id"]
    try:
        if store._collection.get(ids=[rid], include=[])["ids"]:
            store._collection.delete(ids=[rid])
    except Exception: pass
    prior, nbrs = store.compute_prior(commit_title=r.get("commit", ""),
                                      error_lines=r.get("error_lines", []), verbose=False)
    store.upsert_case(run_id=rid, commit_title=r.get("commit", ""), error_lines=r.get("error_lines", []),
                      category=cat(r, "apa"), gt_verdict=av, repo=r.get("repo", ""))
    pr_rank = rank_of(prior, gt_cat); rank1 = pr_rank == 1; buries = pr_rank >= 5
    outcome = "RESCUE" if (av == "WRONG" and rank1) else ("HURT" if (av == "CORRECT" and buries) else "SAME")
    # L3 final family: override to evidence family on rescue, else keep APA family
    l3_b = r["_eb"] if outcome == "RESCUE" else pred_bucket(cat(r, "apa"))
    results.append(dict(run_id=rid, ev_bucket=r["_eb"], apa_verdict=av, prior_top=max(prior, key=prior.get),
                        prior_rank_gt=pr_rank, n_nbrs=len(nbrs), outcome=outcome,
                        apa_correct=(av == "CORRECT"), l3_correct=(l3_b == r["_eb"])))
    if (i + 1) % 50 == 0: print(f"  {i+1}/{len(scor)}", flush=True)

json.dump(results, open("data/l3_evidence_eval.json", "w"), indent=2)
N = len(results)
rank1 = sum(1 for r in results if r["prior_rank_gt"] == 1)
resc = [r for r in results if r["outcome"] == "RESCUE"]; hurt = [r for r in results if r["outcome"] == "HURT"]
l2c = sum(1 for r in results if r["apa_correct"]); l3c = sum(1 for r in results if r["l3_correct"])
no_nbr = sum(1 for r in results if r["n_nbrs"] == 0)
print(f"\n=== FAITHFUL L3 vs EVIDENCE (coarse family, n={N}) ===")
print(f"  evidence family rank-1 in prior: {rank1}/{N} = {rank1/N:.0%}")
print(f"  L2 (APA) correct vs evidence:    {l2c}/{N} = {l2c/N:.0%}")
print(f"  L3 (APA + retrieval):            {l3c}/{N} = {l3c/N:.0%}")
print(f"  RESCUE {len(resc)} : HURT {len(hurt)}   | no neighbour above 0.72 floor: {no_nbr} ({no_nbr/N:.0%})")
