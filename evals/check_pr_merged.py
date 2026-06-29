import json
data = json.load(open("data/benchmark_1000_eig_vs_rpa.json"))
for r in data:
    gt = r.get("ground_truth", {})
    if gt.get("action") == "PR_MERGED":
        print("Repo:", r["repo"])
        print("  GT reasoning:", gt["reasoning"][:200])
        print("  APA:", r["apa_eig"]["prediction"]["category"], "-> judge:", r["apa_eig"]["judge"]["verdict"])
        print("  RPA:", r["rpa"]["prediction"]["category"], "-> judge:", r["rpa"]["judge"]["verdict"])
        print()
