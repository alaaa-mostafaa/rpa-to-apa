import json
from pathlib import Path
from collections import Counter

p = Path("data/benchmark_1000_eig_vs_rpa.json")
results = json.loads(p.read_text())

ns_cases = []
for r in results:
    rpa_v = r.get("rpa", {}).get("judge", {}).get("verdict", "")
    apa_v = r.get("apa_eig", {}).get("judge", {}).get("verdict", "")

    if rpa_v == "NOT_SCORABLE" or apa_v == "NOT_SCORABLE":
        gt = r.get("ground_truth", {})
        apa_reasoning = r.get("apa_eig", {}).get("judge", {}).get("reasoning", "")[:150]
        ns_cases.append({
            "repo": r.get("repo", ""),
            "gt_action": gt.get("action", "MISSING"),
            "gt_method": gt.get("method", "MISSING"),
            "apa_prediction": r.get("apa_eig", {}).get("prediction", {}).get("category", ""),
            "apa_reasoning": apa_reasoning,
        })

print(f"Total processed: {len(results)}")
print(f"Total NOT_SCORABLE: {len(ns_cases)}")
print()

action_counts = Counter(x["gt_action"] for x in ns_cases)
print("=== Ground Truth Action Breakdown ===")
for action, count in action_counts.most_common():
    print(f"  {action}: {count}")

print()
method_counts = Counter(x["gt_method"] for x in ns_cases)
print("=== Classification Method Breakdown ===")
for method, count in method_counts.most_common():
    print(f"  {method}: {count}")

print()
print("=== First 10 sample cases ===")
for r in ns_cases[:10]:
    print(f"  Repo:          {r['repo']}")
    print(f"  GT Action:     {r['gt_action']} | Method: {r['gt_method']}")
    print(f"  APA Predicted: {r['apa_prediction']}")
    print(f"  Judge says:    {r['apa_reasoning']}")
    print()
