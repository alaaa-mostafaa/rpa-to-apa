# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# L3-AUGMENTED remediation: same dual-judge remediation as l4_remediation_dual.py, but the
# agent is given RETRIEVED SIMILAR PAST FAILURES + their known fixes (leave-one-out, no
# leakage of the case's own fix). This is the RAG mechanism LogSage/Bui credit for high
# remediation accuracy. Measures whether L3 lifts remediation above the L2-only baseline.
import json, os, sys, collections, gzip, math
sys.path.insert(0, ".")
os.environ["CI_AGENT_SKIP_ACTION"] = "0"
os.environ["CI_AGENT_CLASSIFY_MODEL"] = "deepseek-reasoner"
os.environ["CI_AGENT_MAX_STEPS"] = "5"
os.environ["CI_AGENT_ACTION_FEWSHOT"] = "1"  # 3-shot grounded fix examples (action-only; classification unchanged)
os.environ["CI_AGENT_DOMAIN"] = "1"          # CI failure->fix domain-knowledge primer
os.environ["CI_AGENT_ACTION_MODEL"] = "deepseek-reasoner"  # stronger model writes the fix
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import (build_agent_graph, _initial_tools_for_event, _fetch_commit_diff,
                           _summarize_patch_files, _fetch_run_history, _failed_step_summary,
                           _focus_log_on_error, _ERROR_FOCUS_RE)
from src.apa.bayesian_tracker import BeliefState
from src.apa.llm_config import make_client
from evals.coarse_eval import substantive_fix_buckets, _load_expert_revisions, pred_bucket
from dataclasses import asdict

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rev = _load_expert_revisions()
ds = {}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            if l.strip():
                r = json.loads(l); ds[r["intake"]["run_id"]] = r

rows = [json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip()]
def has_err(rid):
    ext = ds.get(rid, {}).get("extraction", {})
    txt = "\n".join(e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e, dict) and e.get("text"))
    return bool(_ERROR_FOCUS_RE.search(txt))
scor = []
for r in rows:
    if rev.get(r["run_id"], {}).get("action") == "NOT_SCORABLE": continue
    pr, _ = substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if pr is None or r["run_id"] not in ds: continue
    # Run on ALL 300 scorable cases (consistent with classification). We TAG whether each
    # has a diagnosable error so we can report both the full-300 number and the
    # diagnosable-subset number (cases where remediation is even well-posed).
    r["_has_err"] = has_err(r["run_id"])
    scor.append(r)
scor = scor[:N]
nd = sum(1 for r in scor if r["_has_err"])
print(f"L3-augmented remediation on {len(scor)} cases ({nd} with a diagnosable error)", flush=True)

# ── Build retrieval index over ALL scorable cases (leave-one-out at query time) ──
oai = make_client(provider="openai")
import time
def embed(texts):
    out = []
    for i in range(0, len(texts), 32):   # smaller batches: avoid request timeouts
        batch = texts[i:i+32]
        for attempt in range(4):
            try:
                resp = oai.embeddings.create(model="text-embedding-3-small", input=batch, timeout=60)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  embed batch {i} retry {attempt+1} after {type(e).__name__}", flush=True)
                time.sleep(3 * (attempt + 1))
    return out
def case_text(r):
    errs = " ".join((r.get("error_lines") or [])[:5])
    return f"{r.get('commit','')[:120]} | {errs}"[:500]
print("embedding cases for retrieval...", flush=True)
vecs = embed([case_text(r) for r in scor])
def cos(a, b):
    s = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return s/(na*nb) if na and nb else 0.0
def neighbors(idx, k=3):
    q = vecs[idx]; sims = sorted(((cos(q, vecs[j]), j) for j in range(len(scor)) if j != idx), reverse=True)
    out = []
    for sim, j in sims[:k]:
        rn = scor[j]
        gp, _ = substantive_fix_buckets(rn["ground_truth"].get("reasoning"))
        out.append({"category": gp or "?", "repo": rn["repo"],
                    "fix_reasoning": str(rn["ground_truth"].get("reasoning",""))[:200], "semantic_score": round(sim,3)})
    return out

def run_agent(c, sims):
    event = R._event_from_case(c); kwargs = asdict(event)
    ext = c.get("extraction", {}); excs = ext.get("log_excerpts", [])
    log_texts = [e.get("text","") for e in excs if isinstance(e, dict) and "text" in e]
    error_lines = ext.get("sample_error_lines", [])
    cf, cds = [], {}
    if event.repo and event.commit_sha:
        cf = _fetch_commit_diff(event.repo, event.commit_sha).get("files", []); cds = _summarize_patch_files(cf)
    rh = {}
    if event.repo and event.branch and event.run_number:
        try: rh = _fetch_run_history(event.repo, event.branch, event.workflow or "", event.run_number)
        except Exception: pass
    p = BeliefState()
    st = {"run_event": kwargs, "raw_run": c, "beliefs": dict(p.probabilities), "belief_history": list(p.history),
          "confidence": p.confidence(), "entropy": p.entropy(), "tools_available": _initial_tools_for_event(c, event),
          "tools_called": [], "investigation_log": [], "current_step": 0, "done": False, "error_lines": error_lines,
          "mentioned_files": ext.get("mentioned_files", []), "log_excerpt_texts": log_texts, "changed_files": cf,
          "commit_diff": cds, "failed_step_context": _failed_step_summary(event), "dependency_changes": {},
          "run_history": rh, "similar_failures": sims, "workflow_contents": [], "runner_environment": {}, "pr_context": {},
          "semantic_diff_links": [], "preprocessing_summary": {"top_category": "CODE_REGRESSION"}, "classification": {},
          "api_key": "", "model": os.environ["CI_AGENT_MODEL"], "_next_tool": ""}
    return build_agent_graph().invoke(st).get("classification", {})

CAUSE = """You are a senior engineer judging whether an automated CI-triage tool's recommended fix would resolve a failed build. The developer's actual change is ONE reference (not the only valid fix).
RULES: HELPFUL = a plausible concrete fix addressing the actual error (credit a different-but-valid fix; reverting a breaking commit counts; a SOURCE fix counts even if the error appeared in a TEST file; if the named file is in the developer's changed files that is strong evidence). PARTIAL = right subsystem but incomplete. UNHELPFUL = wrong area (not in changed files and contradicted by the error), gives up ("re-run/flaky"), or too vague. When torn, pick the MORE generous grade.
FAILURE Repo: {repo} | Error: {errors}
TOOL FIX: {rec}
DEVELOPER ACTUALLY DID (reference; note changed files): {gt}
Respond ONLY JSON: {{"verdict":"HELPFUL|PARTIAL|UNHELPFUL","reason":"one sentence"}}"""

def judge(repo, errs, rec, gt):
    j = oai.chat.completions.create(model="gpt-4o", temperature=0.0, max_tokens=120,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":CAUSE.format(repo=repo, errors=errs[:1200], rec=rec[:700], gt=gt[:1100])}])
    try: return json.loads(j.choices[0].message.content).get("verdict")
    except Exception: return "PARSE_ERROR"

def evidence(rid, fb):
    ext = ds.get(rid, {}).get("extraction", {})
    excs = [e.get("text","") for e in (ext.get("log_excerpts") or []) if isinstance(e, dict) and e.get("text")]
    return _focus_log_on_error(excs, 1400).strip() or fb or "(generic exit code)"

out = []
for i, r in enumerate(scor):
    sims = neighbors(i, k=3)
    cls = run_agent(ds[r["run_id"]], sims)
    rec = cls.get("recommended_action","") or "(none)"
    errs = evidence(r["run_id"], "; ".join((r.get("error_lines") or [])[:5]))
    v = judge(r["repo"], errs, rec, str(r["ground_truth"].get("reasoning","")))
    rv = rev.get(r["run_id"])
    gp,_=substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if rv and rv.get("action") in ("CODE","CONFIG","DEPENDENCY"): gp = rv["action"]
    pb = pred_bucket(cls.get("category"))
    out.append(dict(run_id=r["run_id"], repo=r["repo"], category=cls.get("category"),
                    gt_bucket=gp, pred_bucket=pb,
                    cls_correct=bool(pb and gp and pb==gp), recommendation=rec, cause=v,
                    has_err=r["_has_err"], n_sims=len(sims)))
    print(f"[{i+1}/{len(scor)}] {r['repo'][:24]:24} {('OK ' if out[-1]['cls_correct'] else 'X  ')} -> {v}", flush=True)

json.dump(out, open("data/remediation_l3.json","w"), indent=2)
def rep(rows_, title):
    c = collections.Counter(o["cause"] for o in rows_); hp = c.get("HELPFUL",0)+c.get("PARTIAL",0)
    print(f"\n=== {title} (n={len(rows_)}) ===")
    for k,n in c.most_common(): print(f"  {k:10} {n} ({n/max(len(rows_),1):.0%})")
    print(f"  HELPFUL+PARTIAL: {hp}/{len(rows_)} ({hp/max(len(rows_),1):.0%})")
rep(out, "L3-AUGMENTED REMEDIATION — ALL 300")
rep([o for o in out if o["has_err"]], "subset: cases WITH a diagnosable error")

# Deterministic, ungameable LOCALIZATION metric: did the agent's fix name a file the
# developer actually changed? (file-overlap, no LLM judge — the LogSage-comparable metric).
import re as _re
NOISE = {"CHANGELOG.md", "README.md"}
def _files(s):
    return set(f.split("/")[-1] for f in _re.findall(r"[\w./-]+\.\w+", s or "")) - NOISE
loc = tot = useful = 0
for o in out:
    r = rows.get(o["run_id"], {})
    rea = str(r.get("ground_truth", {}).get("reasoning", ""))
    dfs = _files(rea.split("Files:")[-1] if "Files:" in rea else "")
    if not dfs: continue
    tot += 1
    m = bool(dfs & _files(o.get("recommendation", "")))
    o["localized"] = m
    if m: loc += 1
    if m or o["cause"] in ("HELPFUL", "PARTIAL"): useful += 1
json.dump(out, open("data/remediation_l3.json", "w"), indent=2)
print(f"\n=== LOCALIZATION (agent named a developer-changed file; deterministic) ===")
print(f"  {loc}/{tot} = {loc/max(tot,1):.0%}")
print(f"=== USEFUL (resolves [judge] OR localizes [file-overlap]) ===")
print(f"  {useful}/{tot} = {useful/max(tot,1):.0%}")

# Classification (L3-augmented, since retrieval was injected into the agent) from THIS run.
clas = [o for o in out if o["gt_bucket"] and o["pred_bucket"]]
mic = sum(o["cls_correct"] for o in clas) / max(len(clas), 1)
per = collections.defaultdict(lambda: [0, 0])
for o in clas:
    per[o["gt_bucket"]][1] += 1; per[o["gt_bucket"]][0] += o["cls_correct"]
mac = sum(per[b][0]/per[b][1] for b in per) / max(len(per), 1)
print(f"\n=== CLASSIFICATION (L3-augmented, this run) ===")
print(f"  micro={mic:.1%} | macro={mac:.1%} | n={len(clas)}")
for b in per: print(f"    {b}: {per[b][0]}/{per[b][1]} = {per[b][0]/per[b][1]:.0%}")
