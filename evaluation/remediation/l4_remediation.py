# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Remediation-quality evaluation: generate APA's 2-3 sentence fix recommendation
# for each case, then have a HIGHER judge (GPT-4o) rate whether that fix would
# have helped, against the developer's actual fix. Generation uses gpt-4o-mini
# (cheap); judging uses gpt-4o (the high-capability judge).
import json, os, sys, collections
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
from evals.coarse_eval import substantive_fix_buckets, _load_expert_revisions
from src.apa.llm_config import make_client

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GEN_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"
client = make_client(provider="openai")
rev = _load_expert_revisions()

rows = [json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip()]
scor = []
for r in rows:
    if rev.get(r["run_id"], {}).get("action") == "NOT_SCORABLE":
        continue
    pr, _ = substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if pr is None:
        continue
    scor.append(r)
scor = scor[:N]

def cat(r, s): return (r.get(s, {}).get("prediction") or {}).get("category")

GEN_PROMPT = """You are a CI/CD engineer. A pipeline run failed. Based on the diagnosis below, write a SPECIFIC, actionable fix in 2-3 sentences. Name the concrete action (e.g. pin package X to version Y, revert commit, fix the failing assertion in file Z, update the workflow action to v4). Do not say "investigate".

Repo: {repo}
Triggering commit: {commit}
Error lines: {errors}
Diagnosed failure category: {category}
Diagnostic reasoning: {reasoning}

Respond with ONLY the 2-3 sentence fix recommendation."""

JUDGE_PROMPT = """You are a senior engineer reviewing an automated triage tool. A CI run failed; the tool produced a recommended fix. You are told what the developer ACTUALLY did to fix it. Judge whether the tool's recommendation would have led the developer to the correct fix.

FAILURE
  Repo: {repo}
  Commit: {commit}
  Error lines: {errors}

TOOL'S RECOMMENDED FIX:
  {recommendation}

WHAT THE DEVELOPER ACTUALLY DID:
  {gt}

Rate the recommendation:
  HELPFUL    - would have led the developer to substantially the right fix
  PARTIAL    - right direction but missing the key action, or only partially correct
  UNHELPFUL  - wrong direction; would not have helped

Respond ONLY as JSON: {{"verdict": "HELPFUL|PARTIAL|UNHELPFUL", "reason": "one sentence"}}"""

out = []
for i, r in enumerate(scor, 1):
    errs = "; ".join((r.get("error_lines") or [])[:5]) or "(generic exit code)"
    gen = client.chat.completions.create(model=GEN_MODEL, temperature=0.2, max_tokens=160,
        messages=[{"role": "user", "content": GEN_PROMPT.format(
            repo=r["repo"], commit=r.get("commit", "")[:100], errors=errs[:400],
            category=cat(r, "apa"), reasoning=str((r.get("apa", {}).get("prediction") or {}).get("reasoning", ""))[:500])}])
    rec = gen.choices[0].message.content.strip()
    jud = client.chat.completions.create(model=JUDGE_MODEL, temperature=0.0, max_tokens=120,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            repo=r["repo"], commit=r.get("commit", "")[:100], errors=errs[:400],
            recommendation=rec, gt=str(r["ground_truth"].get("reasoning", ""))[:600])}])
    try:
        v = json.loads(jud.choices[0].message.content)
    except Exception:
        v = {"verdict": "PARSE_ERROR", "reason": jud.choices[0].message.content[:80]}
    out.append(dict(run_id=r["run_id"], repo=r["repo"], apa_cat=cat(r, "apa"),
                    recommendation=rec, verdict=v.get("verdict"), reason=v.get("reason")))
    print(f"[{i}/{len(scor)}] {r['repo'][:30]:30} {v.get('verdict')}")

json.dump(out, open("data/remediation_eval.json", "w"), indent=2)
dist = collections.Counter(o["verdict"] for o in out)
print("\n=== Remediation quality (GPT-4o judge) ===")
for k, n in dist.most_common():
    print(f"  {k:10} {n} ({n/len(out):.0%})")
print("\n--- sample ---")
for o in out[:3]:
    print(f"  {o['repo'][:28]} [{o['verdict']}]")
    print(f"    FIX: {o['recommendation'][:140]}")
    print(f"    JUDGE: {o['reason']}")
