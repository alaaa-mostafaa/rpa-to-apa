import json
from pathlib import Path

p = Path("data/benchmark_1000_eig_vs_rpa.json")
results = json.loads(p.read_text())

# Find cases where GT action is valid but verdict is NOT_SCORABLE
print("=== VALID GT but marked NOT_SCORABLE ===")
valid_actions = ("CODE_FIX", "CODE_CHANGE", "PIN_VERSION", "WORKFLOW_FIX", "REVERT")
for r in results:
    gt_action = r.get("ground_truth", {}).get("action", "")
    rpa_v = r.get("rpa", {}).get("judge", {}).get("verdict", "")
    apa_v = r.get("apa_eig", {}).get("judge", {}).get("verdict", "")
    if gt_action in valid_actions and (rpa_v == "NOT_SCORABLE" or apa_v == "NOT_SCORABLE"):
        print("Repo:", r["repo"])
        print("GT:", gt_action, "| RPA verdict:", rpa_v, "| APA verdict:", apa_v)
        print("RPA reasoning:", r["rpa"]["judge"]["reasoning"][:200])
        print("APA reasoning:", r["apa_eig"]["judge"]["reasoning"][:200])
        print()

print()
print("=== EVALUATION_ERROR GT cases ===")
for r in results:
    gt_action = r.get("ground_truth", {}).get("action", "")
    if gt_action == "EVALUATION_ERROR":
        gt_reasoning = r.get("ground_truth", {}).get("reasoning", "")[:200]
        print("Repo:", r["repo"])
        print("GT reasoning:", gt_reasoning)
        print()
