# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
import json, os, sys, collections
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
from evals.coarse_eval import (substantive_fix_buckets, substantive_credit, pred_bucket,
                               BUCKETS, _load_expert_revisions)

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

def cat(r, s): return (r.get(s, {}).get("prediction") or {}).get("category")
def cred(r, s, **kw):
    rv = rev.get(r["run_id"])
    if rv and rv.get("action") in BUCKETS:
        return 1.0 if pred_bucket(cat(r, s)) == rv["action"] else 0.0
    return substantive_credit(cat(r, s), r["ground_truth"].get("reasoning"), **kw) or 0.0

LENS = {"strict": dict(partial=False), "partial": dict(partial=True),
        "multilabel": dict(partial=True, multilabel=True)}
results = {}
for lens, kw in LENS.items():
    per = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for r in scor:
        d = per[r["_pr"]]; d[2] += 1; d[0] += cred(r, "rpa", **kw); d[1] += cred(r, "apa", **kw)
    used = [b for b in BUCKETS if b in per]; ndep = [b for b in used if b != "DEPENDENCY"]
    mac = lambda i, bk: sum(per[b][i] / per[b][2] for b in bk) / len(bk)
    results[lens] = dict(per={b: list(per[b]) for b in per}, macro_r=mac(0, used), macro_a=mac(1, used),
                         exdep_r=mac(0, ndep), exdep_a=mac(1, ndep))
    print(f"{lens:11} macro RPA {results[lens]['macro_r']:.3f} APA {results[lens]['macro_a']:.3f}"
          f" | exdep RPA {results[lens]['exdep_r']:.3f} APA {results[lens]['exdep_a']:.3f}")
P = results["partial"]["per"]
print("per-bucket:", {b: (P[b][2], round(100*P[b][0]/P[b][2]), round(100*P[b][1]/P[b][2])) for b in P})

rpa_dist = collections.Counter(cat(r, "rpa") for r in scor)
apa_dist = collections.Counter(cat(r, "apa") for r in scor)
div = dict(apa_right=0, rpa_right=0, both=0, n=0)
for r in scor:
    if pred_bucket(cat(r, "rpa")) != pred_bucket(cat(r, "apa")):
        div["n"] += 1; ac = cred(r, "apa", partial=False); rc = cred(r, "rpa", partial=False)
        if ac == 1 and rc != 1: div["apa_right"] += 1
        elif rc == 1 and ac != 1: div["rpa_right"] += 1
        else: div["both"] += 1
conf = collections.defaultdict(lambda: collections.Counter())
for r in scor: conf[r["_pr"]][pred_bucket(cat(r, "apa")) or "OTHER"] += 1

labels = json.loads(open("data/evidence_labels_v2.json").read()) if os.path.exists("data/evidence_labels_v2.json") else {}
FINE = {"CODE_REGRESSION":"CODE","QUALITY_VIOLATION":"CODE","TEST_FLAKINESS":"CODE","DEPENDENCY_CONFLICT":"DEPENDENCY","CONFIG_ERROR":"CONFIG","INFRA_INCOMPATIBILITY":"CONFIG","ENV_FLAKINESS":"TRANSIENT","CASCADE_FAILURE":"TRANSIENT","TOOLING_ARTIFACT":"TRANSIENT"}
evb = collections.defaultdict(lambda: [0, 0, 0]); nun = 0
for r in scor:
    eb = FINE.get(labels.get(r["run_id"], {}).get("primary"))
    if eb is None: nun += 1; continue
    d = evb[eb]; d[2] += 1; d[0] += pred_bucket(cat(r, "rpa")) == eb; d[1] += pred_bucket(cat(r, "apa")) == eb
eu = [b for b in BUCKETS if b in evb]
evB = dict(macro_r=sum(evb[b][0]/evb[b][2] for b in eu)/len(eu), macro_a=sum(evb[b][1]/evb[b][2] for b in eu)/len(eu),
           dropped=nun, scorable=sum(d[2] for d in evb.values()))
print(f"ValidatorB macro RPA {evB['macro_r']:.3f} APA {evB['macro_a']:.3f} (n={evB['scorable']}, dropped {nun})")
print("divergence", div)
json.dump(dict(N=len(scor), results=results, rpa_dist=dict(rpa_dist), apa_dist=dict(apa_dist),
               div=div, evB=evB, conf={k: dict(v) for k, v in conf.items()}),
          open("data/eval_numbers.json", "w"), indent=2)

# figures
D = json.load(open("data/eval_numbers.json")); R = D["results"]; RPA = "#E8743B"; APA = "#1F77B4"
plt.rcParams.update({"font.size": 11}); w = 0.36; lenses = ["strict", "partial", "multilabel"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2)); x = np.arange(3)
for ax, (kr, ka, t) in zip(axes, [("macro_r","macro_a","Full macro (all buckets)"), ("exdep_r","exdep_a","Macro, dependency excluded")]):
    rr = [R[l][kr]*100 for l in lenses]; aa = [R[l][ka]*100 for l in lenses]
    ax.bar(x-w/2, rr, w, label="RPA (L1)", color=RPA); ax.bar(x+w/2, aa, w, label="APA (L2)", color=APA)
    for i, (p, q) in enumerate(zip(rr, aa)):
        ax.text(i-w/2, p+1, f"{p:.0f}", ha="center", fontsize=9); ax.text(i+w/2, q+1, f"{q:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["strict","partial","multi-label"]); ax.set_ylim(0,100)
    ax.set_ylabel("Macro accuracy (%)"); ax.set_title(t); ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("images/eval_macro_lenses.png", dpi=150); plt.close()

per = R["partial"]["per"]; bks = [b for b in ["CODE","DEPENDENCY","CONFIG"] if b in per]
rr = [per[b][0]/per[b][2]*100 for b in bks]; aa = [per[b][1]/per[b][2]*100 for b in bks]; x = np.arange(len(bks))
fig, ax = plt.subplots(figsize=(7, 4.3)); ax.bar(x-w/2, rr, w, label="RPA (L1)", color=RPA); ax.bar(x+w/2, aa, w, label="APA (L2)", color=APA)
for i, (p, q) in enumerate(zip(rr, aa)):
    ax.text(i-w/2, p+1, f"{p:.0f}%", ha="center", fontsize=9); ax.text(i+w/2, q+1, f"{q:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={per[b][2]})" for b in bks]); ax.set_ylim(0,105)
ax.set_ylabel("Accuracy (%, partial)"); ax.set_title("Per-category accuracy: RPA vs APA"); ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("images/eval_per_bucket.png", dpi=150); plt.close()

cats = ["CODE_REGRESSION","DEPENDENCY_CONFLICT","CONFIG_ERROR","QUALITY_VIOLATION","TEST_FLAKINESS","INFRA_INCOMPATIBILITY","ENV_FLAKINESS"]
rv2 = [D["rpa_dist"].get(c, 0) for c in cats]; av = [D["apa_dist"].get(c, 0) for c in cats]; x = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(10, 4.3)); ax.bar(x-w/2, rv2, w, label="RPA (L1)", color=RPA); ax.bar(x+w/2, av, w, label="APA (L2)", color=APA)
ax.set_xticks(x); ax.set_xticklabels([c.replace("_","\n") for c in cats], fontsize=8); ax.set_ylabel("# predictions (of 300)")
ax.set_title("Predicted-category distribution: RPA is a near-constant CODE predictor"); ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("images/eval_pred_distribution.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.3)); gA = [R["partial"]["macro_r"]*100, R["partial"]["macro_a"]*100]; gB = [D["evB"]["macro_r"]*100, D["evB"]["macro_a"]*100]; x = np.arange(2)
ax.bar(x-w/2, [gA[0],gB[0]], w, label="RPA (L1)", color=RPA); ax.bar(x+w/2, [gA[1],gB[1]], w, label="APA (L2)", color=APA)
for i, (p, q) in enumerate(zip([gA[0],gB[0]], [gA[1],gB[1]])):
    ax.text(i-w/2, p+1, f"{p:.0f}", ha="center", fontsize=9); ax.text(i+w/2, q+1, f"{q:.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"Validator A\n(dev action, n=300)", f"Validator B\n(evidence, n={D['evB']['scorable']})"])
ax.set_ylim(0,100); ax.set_ylabel("Macro accuracy (%)"); ax.set_title("APA beats RPA under both independent validators"); ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("images/eval_dual_validator.png", dpi=150); plt.close()

order = ["CODE","DEPENDENCY","CONFIG","TRANSIENT","OTHER"]; conf = D["conf"]; gts = [b for b in ["CODE","DEPENDENCY","CONFIG"] if b in conf]
M = np.array([[conf[g].get(p, 0) for p in order] for g in gts], float); Mn = M/M.sum(1, keepdims=True)*100
fig, ax = plt.subplots(figsize=(7, 3.8)); im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=9); ax.set_yticks(range(len(gts))); ax.set_yticklabels(gts)
ax.set_xlabel("APA predicted bucket"); ax.set_ylabel("Ground-truth bucket"); ax.set_title("APA bucket confusion (row-normalized %)")
for i in range(len(gts)):
    for j in range(len(order)):
        if M[i, j] > 0: ax.text(j, i, f"{int(M[i,j])}\n{Mn[i,j]:.0f}%", ha="center", va="center", fontsize=8, color="white" if Mn[i,j] > 50 else "black")
plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.savefig("images/eval_apa_confusion.png", dpi=150); plt.close()

dv = D["div"]; fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(["APA right,\nRPA wrong","RPA right,\nAPA wrong","both wrong"], [dv["apa_right"], dv["rpa_right"], dv["both"]], color=[APA, RPA, "#999"])
for i, v in enumerate([dv["apa_right"], dv["rpa_right"], dv["both"]]): ax.text(i, v+0.5, str(v), ha="center")
ax.set_ylabel(f"# cases (of {dv['n']} disagreements)"); ax.set_title("When RPA and APA disagree, who is right?"); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("images/eval_divergence.png", dpi=150); plt.close()
print("regenerated all 6 figures + eval_numbers.json")
