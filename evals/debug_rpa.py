import json
from pathlib import Path
from collections import Counter

p = Path("data/benchmark_1000_eig_vs_rpa.json")
results = json.loads(p.read_text())

scorable = []
for r in results:
    rpa_v = r.get("rpa", {}).get("judge", {}).get("verdict", "")
    apa_v = r.get("apa_eig", {}).get("judge", {}).get("verdict", "")
    if rpa_v in ("EVALUATION_ERROR", "NOT_SCORABLE") or apa_v in ("EVALUATION_ERROR", "NOT_SCORABLE"):
        continue
    scorable.append(r)

print(f"Scorable cases: {len(scorable)}")
print()

rpa_verdicts = Counter(r["rpa"]["judge"]["verdict"] for r in scorable)
apa_verdicts = Counter(r["apa_eig"]["judge"]["verdict"] for r in scorable)
print("RPA verdicts:", dict(rpa_verdicts))
print("APA verdicts:", dict(apa_verdicts))
print()

print("=== RPA WRONG cases: what did RPA predict vs what was the fix? ===")
for r in scorable:
    if r["rpa"]["judge"]["verdict"] == "WRONG":
        rpa_cat = r.get("rpa", {}).get("prediction", {}).get("category", "?")
        gt_action = r.get("ground_truth", {}).get("action", "?")
        apa_v = r["apa_eig"]["judge"]["verdict"]
        apa_cat = r.get("apa_eig", {}).get("prediction", {}).get("category", "?")
        print(f"  Repo: {r['repo']}")
        print(f"  RPA predicted: {rpa_cat} | GT action: {gt_action}")
        print(f"  APA predicted: {apa_cat} | APA verdict: {apa_v}")
        print(f"  RPA reasoning: {r['rpa']['judge']['reasoning'][:150]}")
        print()
