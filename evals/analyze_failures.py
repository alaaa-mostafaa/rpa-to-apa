"""Analyze APA failure patterns from the benchmark results — v2 with correct field paths."""
import json

data = json.load(open("data/benchmark_1000_eig_vs_rpa.json"))

apa_correct = []
apa_wrong = []
apa_partial = []
apa_not_scorable = []

for r in data:
    apa_v = r["apa_eig"]["judge"]["verdict"]
    rpa_v = r["rpa"]["judge"]["verdict"]
    repo = r.get("repo", "")
    apa_cat = r["apa_eig"]["prediction"]["category"]
    rpa_cat = r["rpa"]["prediction"]["category"]
    gt = r.get("ground_truth", {})
    gt_action = gt.get("action", "?")
    gt_reasoning = gt.get("reasoning", "")[:120]
    apa_judge_reasoning = r["apa_eig"]["judge"].get("reasoning", "")[:150]
    apa_pred_reasoning = r["apa_eig"]["prediction"].get("reasoning", "")[:120]

    entry = {
        "repo": repo,
        "apa_cat": apa_cat,
        "rpa_cat": rpa_cat,
        "gt_action": gt_action,
        "gt_reasoning": gt_reasoning,
        "apa_judge_reasoning": apa_judge_reasoning,
        "apa_pred_reasoning": apa_pred_reasoning,
        "rpa_v": rpa_v,
    }

    if apa_v == "CORRECT":
        apa_correct.append(entry)
    elif apa_v == "WRONG":
        apa_wrong.append(entry)
    elif apa_v == "PARTIAL":
        apa_partial.append(entry)
    else:
        apa_not_scorable.append(entry)

print(f"=== APA VERDICT BREAKDOWN (n={len(data)}) ===")
print(f"CORRECT:      {len(apa_correct)}")
print(f"WRONG:        {len(apa_wrong)}")
print(f"PARTIAL:      {len(apa_partial)}")
print(f"NOT_SCORABLE: {len(apa_not_scorable)}")
print()

# Category distribution of WRONG cases
from collections import Counter
wrong_apa_cats = Counter(e["apa_cat"] for e in apa_wrong)
wrong_gt_actions = Counter(e["gt_action"] for e in apa_wrong)
print("=== WRONG: What APA classified as ===")
for cat, cnt in wrong_apa_cats.most_common():
    print(f"  {cat}: {cnt}")
print()
print("=== WRONG: What the developer actually did (GT action) ===")
for cat, cnt in wrong_gt_actions.most_common():
    print(f"  {cat}: {cnt}")
print()

print("=" * 80)
print("=== APA WRONG CASES (full detail) ===")
print("=" * 80)
for i, e in enumerate(apa_wrong, 1):
    print(f"\n--- WRONG #{i}: {e['repo']} ---")
    print(f"  APA classified:   {e['apa_cat']}")
    print(f"  RPA classified:   {e['rpa_cat']} (RPA verdict: {e['rpa_v']})")
    print(f"  GT dev action:    {e['gt_action']}")
    print(f"  GT reasoning:     {e['gt_reasoning']}")
    print(f"  APA reasoning:    {e['apa_pred_reasoning']}")
    print(f"  Judge reasoning:  {e['apa_judge_reasoning']}")

print()
print("=" * 80)
print("=== MISMATCH PATTERN ANALYSIS ===")
print("=" * 80)
# What are the specific APA_cat -> GT_action mismatches?
mismatch_pairs = Counter((e["apa_cat"], e["gt_action"]) for e in apa_wrong)
print("\nAPA Category -> GT Action (count):")
for (apa, gt), cnt in mismatch_pairs.most_common():
    print(f"  {apa} -> {gt}: {cnt}")

# Key stat: if PARTIAL counted as CORRECT, what would % be?
scorable = [r for r in data 
            if r["rpa"]["judge"]["verdict"] not in ("NOT_SCORABLE", "EVALUATION_ERROR")
            and r["apa_eig"]["judge"]["verdict"] not in ("NOT_SCORABLE", "EVALUATION_ERROR")]
strict = sum(1 for r in scorable if r["apa_eig"]["judge"]["verdict"] == "CORRECT")
lenient = sum(1 for r in scorable if r["apa_eig"]["judge"]["verdict"] in ("CORRECT", "PARTIAL"))
rpa_strict = sum(1 for r in scorable if r["rpa"]["judge"]["verdict"] == "CORRECT")
print(f"\n=== ACCURACY SUMMARY ===")
print(f"Scorable: {len(scorable)}")
print(f"APA Strict (CORRECT only):      {strict}/{len(scorable)} = {strict/len(scorable)*100:.1f}%")
print(f"APA Lenient (CORRECT+PARTIAL):  {lenient}/{len(scorable)} = {lenient/len(scorable)*100:.1f}%")
print(f"RPA Strict (CORRECT only):      {rpa_strict}/{len(scorable)} = {rpa_strict/len(scorable)*100:.1f}%")

# NOT_SCORABLE breakdown
print(f"\n=== NOT_SCORABLE ANALYSIS ===")
print(f"Total NOT_SCORABLE: {len(apa_not_scorable)}")
ns_gt = Counter(e["gt_action"] for e in apa_not_scorable)
print("GT actions in NOT_SCORABLE cases:")
for cat, cnt in ns_gt.most_common():
    print(f"  {cat}: {cnt}")
