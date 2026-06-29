# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Re-run the L3 (retrieval-prior) evaluation on the NEW 250-case fixed-extractor
# corpus (data/eval_big100.jsonl), reusing the stored L2/APA predictions.
# Only OpenAI embeddings are used (no agent re-run, no DeepSeek). Leave-one-out.
import json, os, sys, collections
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
from evals.coarse_eval import (substantive_fix_buckets, substantive_credit, pred_bucket,
                               BUCKETS, _load_expert_revisions)
from src.apa.chroma_case_store import ChromaCaseStore

rev = _load_expert_revisions()
rows = [json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip()]
scor = []
for r in rows:
    rv = rev.get(r["run_id"])
    if rv and rv.get("action") == "NOT_SCORABLE":
        continue
    pr, t = substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if rv and rv.get("action") in BUCKETS:
        pr = rv["action"]
    if pr is None:
        continue
    r["_pr"] = pr; scor.append(r)
scor = scor[:300]
print(f"L3 re-run on {len(scor)} cases")

def cat(r, s): return (r.get(s, {}).get("prediction") or {}).get("category")
BUCKET_TO_CAT = {"CODE": "CODE_REGRESSION", "DEPENDENCY": "DEPENDENCY_CONFLICT", "CONFIG": "CONFIG_ERROR"}

def apa_verdict(r):
    c = substantive_credit(cat(r, "apa"), r["ground_truth"].get("reasoning"), partial=True)
    return "CORRECT" if c == 1.0 else ("PARTIAL" if c == 0.5 else "WRONG")

# fresh store
import shutil
CH = "data/chroma_l3new"
shutil.rmtree(CH, ignore_errors=True)
store = ChromaCaseStore(path=CH)
for r in scor:
    store.upsert_case(run_id=r["run_id"], commit_title=r.get("commit", ""),
                      error_lines=r.get("error_lines", []), category=cat(r, "apa"),
                      gt_verdict=apa_verdict(r), repo=r.get("repo", ""))
print(f"store populated: {store.count()} cases")

def rank_of(prior, c):
    return sorted(prior, key=prior.get, reverse=True).index(c) + 1 if c in prior else len(prior)

results = []
for i, r in enumerate(scor):
    gt_cat = BUCKET_TO_CAT[r["_pr"]]; av = apa_verdict(r)
    rid = r["run_id"]
    try:
        if store._collection.get(ids=[rid], include=[])["ids"]:
            store._collection.delete(ids=[rid])
    except Exception:
        pass
    prior, nbrs = store.compute_prior(commit_title=r.get("commit", ""),
                                      error_lines=r.get("error_lines", []), verbose=False)
    store.upsert_case(run_id=rid, commit_title=r.get("commit", ""), error_lines=r.get("error_lines", []),
                      category=cat(r, "apa"), gt_verdict=av, repo=r.get("repo", ""))
    pr_rank = rank_of(prior, gt_cat)
    rank1 = pr_rank == 1; buries = pr_rank >= 5
    if av in ("WRONG", "PARTIAL") and rank1:
        outcome = "RESCUE"
    elif av == "CORRECT" and buries:
        outcome = "HURT"
    else:
        outcome = "SAME"
    results.append(dict(run_id=rid, gt_cat=gt_cat, apa_verdict=av,
                        prior_top=max(prior, key=prior.get), prior_rank_gt=pr_rank,
                        n_nbrs=len(nbrs), outcome=outcome))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(scor)}")

json.dump(results, open("data/l3_retrieval_eval_new.json", "w"), indent=2)
N = len(results)
rank1 = sum(1 for r in results if r["prior_rank_gt"] == 1)
rank2 = sum(1 for r in results if r["prior_rank_gt"] == 2)
rescues = [r for r in results if r["outcome"] == "RESCUE"]
hurts = [r for r in results if r["outcome"] == "HURT"]
same = [r for r in results if r["outcome"] == "SAME"]
strict_resc = sum(1 for r in rescues if r["apa_verdict"] == "WRONG")
no_nbr = sum(1 for r in results if r["n_nbrs"] == 0)
print(f"\n=== L3 RESULTS (new {N}-case corpus) ===")
print(f"GT category rank-1 in prior: {rank1}/{N} ({rank1/N:.1%}); rank-2: {rank2}")
print(f"RESCUE {len(rescues)} (strict {strict_resc}, partial {len(rescues)-strict_resc}) | HURT {len(hurts)} | SAME {len(same)}")
print(f"RESCUE:HURT = {len(rescues)}:{len(hurts)}")
print(f"cases with no neighbour above floor: {no_nbr} ({no_nbr/N:.1%})")

# ---- figure 1: rank distribution by L2 verdict ----
maxr = max(r["prior_rank_gt"] for r in results)
ranks = list(range(1, min(maxr, 9) + 1))
corr = [sum(1 for r in results if r["prior_rank_gt"] == k and r["apa_verdict"] == "CORRECT") for k in ranks]
part = [sum(1 for r in results if r["prior_rank_gt"] == k and r["apa_verdict"] == "PARTIAL") for k in ranks]
wrong = [sum(1 for r in results if r["prior_rank_gt"] == k and r["apa_verdict"] == "WRONG") for k in ranks]
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(9, 4.3))
ax.bar(ranks, corr, label="L2 already CORRECT", color="#1F77B4")
ax.bar(ranks, part, bottom=corr, label="L2 PARTIAL", color="#9ecae1")
ax.bar(ranks, wrong, bottom=[c+p for c, p in zip(corr, part)], label="L2 WRONG (rescue if rank 1)", color="#E8743B")
ax.set_xlabel("Rank of ground-truth category in retrieval prior"); ax.set_ylabel("# cases")
ax.set_title("Retrieval-prior rank of the correct category, by L2 verdict"); ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("images/fig_retrieval_rank_dist.png", dpi=150); plt.close()

# ---- figure 2: rescue/hurt ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(["SAME", "RESCUE", "HURT"], [len(same), len(rescues), len(hurts)], color=["#999", "#2ca02c", "#d62728"])
for i, v in enumerate([len(same), len(rescues), len(hurts)]):
    axes[0].text(i, v+1, f"{v}\n{v/N:.0%}", ha="center", fontsize=9)
axes[0].set_ylabel("# cases"); axes[0].set_title("L3 outcome distribution"); axes[0].grid(axis="y", alpha=.3)
axes[1].bar(["RESCUE\n(strict)", "RESCUE\n(partial)", "HURT"], [strict_resc, len(rescues)-strict_resc, len(hurts)],
            color=["#2ca02c", "#98df8a", "#d62728"])
for i, v in enumerate([strict_resc, len(rescues)-strict_resc, len(hurts)]):
    axes[1].text(i, v+0.3, str(v), ha="center", fontsize=9)
axes[1].set_ylabel("# cases"); axes[1].set_title("RESCUE breakdown vs HURT"); axes[1].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("images/fig_rescue_hurt.png", dpi=150); plt.close()
print("regenerated fig_retrieval_rank_dist.png and fig_rescue_hurt.png")
