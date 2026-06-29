# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Score each APA fix recommendation on the five actionability qualities of
# Valenzuela-Toledo et al. (clarity, actionable guidance, specificity, contextual
# relevance, conciseness). Each quality is a yes/no judgement; we report the percentage
# of recommendations that satisfy each. GPT-4o judge; input is the short recommendation
# plus the error lines (no full log), so the run is cheap. Resumable.
import json, os, sys, collections
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
from src.apa.llm_config import make_client

recs = json.load(open("data/remediation_l3.json"))
rows = {json.loads(l)["run_id"]: json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip()}

oai = make_client(provider="openai")
JUDGE = """You are evaluating the ACTIONABILITY of an automated CI/CD failure fix recommendation,
using five qualities from research on what developers find useful. Judge each as true/false.

1. clarity: easy to understand, no unnecessary jargon.
2. actionable_guidance: gives specific steps the developer can directly apply.
3. specificity: tailored to THIS failure, not generic (e.g. names the actual library, file, or error,
   rather than "there is a conflict in the project").
4. contextual_relevance: includes relevant context such as a file name, line, dependency, setting, or doc link.
5. conciseness: brief but informative; does NOT pad with obvious filler like "edit the file" or
   "commit your changes" that a developer already knows.

ERROR LINES:
{errors}

RECOMMENDATION:
{rec}

Respond ONLY JSON with the five booleans:
{{"clarity":true/false,"actionable_guidance":true/false,"specificity":true/false,"contextual_relevance":true/false,"conciseness":true/false}}"""

CK = "data/actionability_results.json"
out = []; done = set()
if os.path.exists(CK):
    out = json.load(open(CK)); done = {o["run_id"] for o in out}
    print(f"resuming: {len(done)} done", flush=True)

QUAL = ["clarity","actionable_guidance","specificity","contextual_relevance","conciseness"]
for i, r in enumerate(recs, 1):
    rid = r["run_id"]
    if rid in done: continue
    rec = (r.get("recommendation") or "").strip()
    if not rec or rec == "(none)":
        continue
    errs = "; ".join((rows.get(rid, {}).get("error_lines") or [])[:6])[:800]
    try:
        j = oai.chat.completions.create(model="gpt-4o", temperature=0.0, max_tokens=120,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": JUDGE.format(errors=errs or "(none)", rec=rec[:600])}])
        v = json.loads(j.choices[0].message.content)
    except Exception as e:
        print(f"STOPPED at [{i}/{len(recs)}] {type(e).__name__}: saving {len(out)}", flush=True)
        json.dump(out, open(CK, "w"), indent=1); raise SystemExit(0)
    out.append(dict(run_id=rid, **{q: bool(v.get(q)) for q in QUAL}))
    if i % 25 == 0:
        print(f"  {i}/{len(recs)}", flush=True); json.dump(out, open(CK, "w"), indent=1)

json.dump(out, open(CK, "w"), indent=1)
n = len(out)
print(f"\n=== ACTIONABILITY OF APA FIX RECOMMENDATIONS (n={n}) ===")
for q in QUAL:
    c = sum(1 for o in out if o[q])
    print(f"  {q:22} {c}/{n} = {c/n:.0%}")
allfive = sum(1 for o in out if all(o[q] for q in QUAL))
print(f"  ALL FIVE satisfied      {allfive}/{n} = {allfive/n:.0%}")
