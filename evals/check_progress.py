import json
from pathlib import Path
from collections import Counter

p = Path("data/benchmark_1000_eig_vs_rpa.json")
if not p.exists():
    print("No results file yet.")
else:
    results = json.loads(p.read_text())
    scorable = 0
    rpa_correct = 0
    apa_correct = 0
    apa_partial = 0
    not_scorable = 0
    eval_errors = 0

    for r in results:
        rpa_v = r.get("rpa", {}).get("judge", {}).get("verdict", "")
        apa_v = r.get("apa_eig", {}).get("judge", {}).get("verdict", "")
        if rpa_v == "EVALUATION_ERROR" or apa_v == "EVALUATION_ERROR":
            eval_errors += 1
        elif rpa_v == "NOT_SCORABLE" or apa_v == "NOT_SCORABLE":
            not_scorable += 1
        else:
            scorable += 1
            if rpa_v == "CORRECT": rpa_correct += 1
            if apa_v == "CORRECT": apa_correct += 1
            if apa_v == "PARTIAL": apa_partial += 1

    print(f"Total processed:   {len(results)}")
    print(f"Scorable:          {scorable}")
    print(f"NOT_SCORABLE:      {not_scorable}")
    print(f"EVALUATION_ERROR:  {eval_errors}")
    print()
    if scorable:
        print(f"RPA Accuracy:              {rpa_correct/scorable*100:.1f}% ({rpa_correct}/{scorable})")
        print(f"APA Accuracy (strict):     {apa_correct/scorable*100:.1f}% ({apa_correct}/{scorable})")
        print(f"APA (strict+partial):      {(apa_correct+apa_partial)/scorable*100:.1f}% ({apa_correct+apa_partial}/{scorable})")
    else:
        print("No scorable cases yet.")

    # GT action breakdown for NOT_SCORABLE
    ns_gt = Counter()
    for r in results:
        rpa_v = r.get("rpa", {}).get("judge", {}).get("verdict", "")
        apa_v = r.get("apa_eig", {}).get("judge", {}).get("verdict", "")
        if rpa_v == "NOT_SCORABLE" or apa_v == "NOT_SCORABLE":
            ns_gt[r.get("ground_truth", {}).get("action", "MISSING")] += 1
    print()
    print("NOT_SCORABLE GT breakdown:")
    for k, v in ns_gt.most_common():
        print(f"  {k}: {v}")
