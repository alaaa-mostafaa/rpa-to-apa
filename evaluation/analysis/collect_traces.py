# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Instrumented run: diagnose a sample of cases with the full APA agent, capturing the
# investigation trace per case (tools called, steps, belief-entropy trajectory, stop reason),
# plus the developer-fix bucket for correctness. Writes data/agent_traces.json.
import json, os, sys, math, gzip, time
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
# Match run_eval_500.py EXACTLY (the harness that generated eval_big100): provider set so
# the bare model ids disambiguate. Prefixed ids without LLM_PROVIDER silently fail the
# deep_log likelihood call -> uniform -> posterior == CODE-peaked prior -> CODE collapse.
os.environ.setdefault("LLM_PROVIDER", "deepseek")
os.environ["CI_AGENT_MODEL"] = os.environ.get("CI_AGENT_MODEL", "deepseek-chat")
os.environ["CI_AGENT_CLASSIFY_MODEL"] = os.environ.get("CI_AGENT_CLASSIFY_MODEL", "deepseek-reasoner")
os.environ["CI_AGENT_SECONDARY_MODEL"] = os.environ.get("CI_AGENT_SECONDARY_MODEL", "deepseek-chat")

from src.apa.framework import _state_from_case
from src.apa.agent import build_agent_graph
from evals.coarse_eval import substantive_fix_buckets, pred_bucket, _load_expert_revisions

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rev = _load_expert_revisions()

ds = {}
for f in ("data/dataset_remote_250.jsonl.gz","data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz","data/dataset_topup.jsonl.gz"):
    if os.path.exists(f):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            if l.strip(): r = json.loads(l); ds[r["intake"]["run_id"]] = r
rows = {json.loads(l)["run_id"]: json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip()}

# scorable cases (developer-fix bucket known), keep dataset record available
scor = []
for rid, r in rows.items():
    if rev.get(rid, {}).get("action") == "NOT_SCORABLE": continue
    pr, _ = substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if rev.get(rid, {}).get("action") in ("CODE", "CONFIG", "DEPENDENCY"): pr = rev[rid]["action"]
    if pr is None or rid not in ds: continue
    scor.append((rid, pr))
scor = scor[:N]
print(f"instrumented run on {len(scor)} cases", flush=True)

def H(b):
    ps = [v for v in b.values() if isinstance(v, (int, float)) and v > 0]
    return -sum(p*math.log2(p) for p in ps)

CKPT = "data/agent_traces.json"
out = []; done = set()
if os.path.exists(CKPT):
    out = json.load(open(CKPT)); done = {o["run_id"] for o in out}
    print(f"resuming: {len(done)} done", flush=True)

graph = build_agent_graph()
for i, (rid, gt_bucket) in enumerate(scor, 1):
    if rid in done: continue
    try:
        st = _state_from_case(ds[rid], os.environ["CI_AGENT_MODEL"])
        final = graph.invoke(st)
    except Exception as e:
        print(f"[{i}] {rid[:40]} ERR {type(e).__name__}: {str(e)[:60]}", flush=True); continue
    cl = final.get("classification", {}) or {}
    bh = [s for s in (final.get("belief_history") or []) if isinstance(s, dict)]
    signals = [s.get("signal") for s in bh]                       # ordered tool/signal sequence
    traj = [round(s["entropy"], 3) for s in bh if isinstance(s.get("entropy"), (int, float))]
    gains = [round(s.get("information_gain", 0.0), 4) for s in bh]
    tools = [s for s in signals if s and s != "deep_log_analysis"]  # escalation tools beyond the log
    steps = len(bh)                                                # belief updates incl. mandatory log
    e_end = traj[-1] if traj else None
    stop = "entropy_threshold" if (e_end is not None and e_end < 1.0) else "budget_or_exhausted"
    pb = pred_bucket(cl.get("category"))
    out.append(dict(
        run_id=rid, repo=rows[rid]["repo"], gt_bucket=gt_bucket,
        category=cl.get("category"), pred_bucket=pb, correct=bool(pb == gt_bucket),
        confidence=cl.get("confidence"), steps=steps, n_tools=len(tools),
        tools=tools, signals=signals, info_gains=gains, stop=stop, entropy_traj=traj,
        entropy_start=traj[0] if traj else None, entropy_end=e_end,
    ))
    print(f"[{i}/{len(scor)}] {rows[rid]['repo'][:22]:22} {str(cl.get('category')):18} "
          f"steps={steps} tools={tools} E:{traj[0] if traj else '?'}->{e_end} {'OK' if pb==gt_bucket else 'x'}", flush=True)
    if len(out) % 10 == 0: json.dump(out, open(CKPT, "w"), indent=1)

json.dump(out, open(CKPT, "w"), indent=1)
print(f"\nwrote {len(out)} traces to {CKPT}", flush=True)
