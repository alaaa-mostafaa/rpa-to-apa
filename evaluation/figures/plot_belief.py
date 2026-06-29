# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
import json
import matplotlib.pyplot as plt

with open("data/benchmark_final_eval.json", "r") as f:
    data = json.load(f)

candidates = []
for d in data:
    apa_key = "apa_eig" if "apa_eig" in d else ("apa_hybrid" if "apa_hybrid" in d else None)
    if apa_key and "belief_history" in d[apa_key]:
        history = d[apa_key]["belief_history"]
        judge = d[apa_key].get("judge", {})
        verdict = judge.get("verdict", "")
        
        # We want an ambiguous start, and confident end
        if len(history) >= 2 and verdict in ("CORRECT", "PARTIAL_MATCH"):
            if "beliefs" in history[0] and "beliefs" in history[-1]:
                init_max = max(history[0]["beliefs"].values())
                final_max = max(history[-1]["beliefs"].values())
                if init_max < 0.6 and final_max > 0.75:
                    tools = d[apa_key].get("tools_called", [])
                    candidates.append({
                        "repo": d["repo"],
                        "steps": len(history),
                        "init_max": init_max,
                        "final_max": final_max,
                        "diff": final_max - init_max,
                        "tools": tools,
                        "history": history
                    })

# Sort by number of steps (we want ~3-4 steps) then diff
candidates.sort(key=lambda x: (abs(x["steps"] - 4), -x["diff"]))

print(f"Found {len(candidates)} candidates")
for c in candidates[:10]:
    print(f"{c['repo']} | steps: {c['steps']} | init: {c['init_max']:.2f} | final: {c['final_max']:.2f} | tools: {c['tools']}")

if candidates:
    target = candidates[0]
    history = target["history"]
    
    # categories to plot (maybe just top 4 to avoid clutter)
    all_cats = set()
    for h in history:
        for k in h["beliefs"]:
            all_cats.add(k)
            
    # get top categories from final step
    final_beliefs = history[-1]["beliefs"]
    top_cats = sorted(all_cats, key=lambda k: final_beliefs.get(k, 0), reverse=True)[:5]
    
    plt.figure(figsize=(10, 6))
    
    steps = list(range(len(history)))
    
    for cat in top_cats:
        probs = [h["beliefs"].get(cat, 0) for h in history]
        linewidth = 3 if cat == top_cats[0] else 1.5
        plt.plot(steps, probs, label=cat, marker='o', linewidth=linewidth)
        
    # Formatting
    plt.title(f"Bayesian Belief Evolution ({target['repo']})")
    plt.xlabel("Investigation Steps")
    plt.ylabel("Probability")
    plt.ylim(0, 1.05)
    plt.xticks(steps, [f"Init" if i == 0 else f"Step {i}\n{target['tools'][i-1]}" if i-1 < len(target['tools']) else f"Step {i}" for i in steps], rotation=45, ha='right')
    plt.legend(title="Failure Category")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/belief_evolution.png", dpi=300)
    print("Saved plot to images/belief_evolution.png")
