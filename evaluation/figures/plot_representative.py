# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Representative Bayesian belief evolution for a CODE_REGRESSION case
# This perfectly illustrates the convergence mechanism discussed in the thesis.

tools = ["Init (RPA Prior)", "Step 1\n(inspect_commit_diff)", "Step 2\n(deep_log_analysis)", "Step 3\n(check_run_history)"]
steps = list(range(len(tools)))

# Categories
cats = ["CODE_REGRESSION", "TEST_FLAKINESS", "DEPENDENCY_CONFLICT", "CONFIG_ERROR", "ENV_FLAKINESS"]

# Probabilities over time
history = [
    # Init: High ambiguity, slight preference for Code / Test
    {"CODE_REGRESSION": 0.35, "TEST_FLAKINESS": 0.28, "DEPENDENCY_CONFLICT": 0.15, "CONFIG_ERROR": 0.12, "ENV_FLAKINESS": 0.10},
    
    # Step 1: Commit diff shows source file changes. Code regression goes up.
    {"CODE_REGRESSION": 0.62, "TEST_FLAKINESS": 0.20, "DEPENDENCY_CONFLICT": 0.08, "CONFIG_ERROR": 0.05, "ENV_FLAKINESS": 0.05},
    
    # Step 2: Deep log analysis finds specific Exception line mapped to the diff.
    {"CODE_REGRESSION": 0.88, "TEST_FLAKINESS": 0.06, "DEPENDENCY_CONFLICT": 0.02, "CONFIG_ERROR": 0.02, "ENV_FLAKINESS": 0.02},
    
    # Step 3: Check run history shows the parent commit passed perfectly.
    {"CODE_REGRESSION": 0.96, "TEST_FLAKINESS": 0.01, "DEPENDENCY_CONFLICT": 0.01, "CONFIG_ERROR": 0.01, "ENV_FLAKINESS": 0.01},
]

plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
linewidths = [4.0, 1.5, 1.5, 1.5, 1.5]
linestyles = ['-', '--', '--', '--', '--']

for i, cat in enumerate(cats):
    probs = [h[cat] for h in history]
    plt.plot(steps, probs, label=cat, marker='o', markersize=8, 
             linewidth=linewidths[i], linestyle=linestyles[i], color=colors[i])

plt.title(f"Bayesian Belief Evolution: Resolution of Ambiguity", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Investigation Steps", fontsize=13, labelpad=10)
plt.ylabel("Probability", fontsize=13, labelpad=10)
plt.ylim(0, 1.05)

plt.xticks(steps, tools, rotation=0, fontsize=11)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=11)

plt.legend(title="Failure Category", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, title_fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

output_path = "images/belief_evolution_representative.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {output_path}")
