# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
"""
generate_creative_figures.py
────────────────────────────
Generates visually compelling methodology chapter figures.
All figures explain the SYSTEM (not evaluation results).
Saves PNGs to images/ at 300 dpi.

Figures:
  A. images/fig_belief_heatmap.png       – Belief state as animated heatmap grid
  B. images/fig_pipeline_flow.png        – RPA→APA pipeline as a styled flow diagram
  C. images/fig_entropy_funnel.png       – Entropy/confidence convergence with shaded zones
  D. images/fig_tool_radar.png           – Radar chart of tool capabilities across failure types
  E. images/fig_signal_cascade.png       – RPA 7-signal waterfall with information gain bars
"""

import math, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

os.makedirs("images", exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "blue":    "#2563EB",
    "lblue":   "#93C5FD",
    "orange":  "#EA580C",
    "lorange": "#FED7AA",
    "green":   "#16A34A",
    "lgreen":  "#BBF7D0",
    "red":     "#DC2626",
    "purple":  "#7C3AED",
    "lpurple": "#DDD6FE",
    "teal":    "#0891B2",
    "gray":    "#64748B",
    "lgray":   "#E2E8F0",
    "dark":    "#1E293B",
    "white":   "#FFFFFF",
    "yellow":  "#CA8A04",
}

CATEGORIES_14 = [
    "CODE_REGRESSION", "DEPENDENCY_CONFLICT", "CONFIG_ERROR",
    "ENV_FLAKINESS", "TEST_FLAKINESS", "TOOLING_ARTIFACT",
    "CASCADE_FAILURE", "INFRA_INCOMPATIBILITY", "NETWORK_TRANSIENT",
    "OOM_KILL", "POD_CRASH", "IMAGE_PULL_BACKOFF",
    "SCHEDULING_ERROR", "UNKNOWN",
]
N_CATS = len(CATEGORIES_14)


# ─────────────────────────────────────────────────────────────────────────────
# Figure A – Belief State Heatmap Grid
# Shows the full 14-category distribution as a colour heatmap across 7 time steps.
# Much more information-dense and visually striking than a line plot.
# ─────────────────────────────────────────────────────────────────────────────

def figA_belief_heatmap():
    """
    2D heatmap: rows = 14 failure categories, columns = investigation steps.
    Colour intensity = probability. The winning category glows.
    """

    # Belief history for a DEPENDENCY_CONFLICT case (dependabot branch, requirements.txt bump)
    # Constructed to be internally consistent with the Bayesian update math.
    # Steps: Init → branch_type → jobs_failed → commit_msg → detection_mode → [APA] inspect_commit_diff → inspect_dep_changes
    cat_idx = {c: i for i, c in enumerate(CATEGORIES_14)}
    n_steps = 7
    step_labels = [
        "Init\n(uniform)",
        "branch_type\n(dependabot)",
        "jobs_failed\n(all fail)",
        "commit_msg\n(\"Bump…\")",
        "detect_mode\n(fallback)",
        "commit_diff\n[APA-1]",
        "dep_changes\n[APA-2]",
    ]
    rpa_boundary = 4  # last RPA step index (0-based)

    history = np.zeros((N_CATS, n_steps))
    def set_step(col, probs: dict):
        total = sum(probs.values())
        for cat, p in probs.items():
            history[cat_idx[cat], col] = p / total

    set_step(0, {c: 1/N_CATS for c in CATEGORIES_14})

    set_step(1, {
        "DEPENDENCY_CONFLICT": 0.22, "CONFIG_ERROR": 0.10,
        "CODE_REGRESSION": 0.08, "INFRA_INCOMPATIBILITY": 0.07,
        "ENV_FLAKINESS": 0.07, "TEST_FLAKINESS": 0.07,
        "TOOLING_ARTIFACT": 0.07, "CASCADE_FAILURE": 0.07,
        "NETWORK_TRANSIENT": 0.06, "UNKNOWN": 0.06,
        "OOM_KILL": 0.04, "POD_CRASH": 0.03, "IMAGE_PULL_BACKOFF": 0.03, "SCHEDULING_ERROR": 0.03,
    })

    set_step(2, {  # all jobs fail → not informative for dep conflict vs config
        "DEPENDENCY_CONFLICT": 0.24, "CONFIG_ERROR": 0.12,
        "INFRA_INCOMPATIBILITY": 0.09, "CODE_REGRESSION": 0.08,
        "ENV_FLAKINESS": 0.07, "TEST_FLAKINESS": 0.06,
        "TOOLING_ARTIFACT": 0.06, "CASCADE_FAILURE": 0.07,
        "NETWORK_TRANSIENT": 0.06, "UNKNOWN": 0.06,
        "OOM_KILL": 0.04, "POD_CRASH": 0.02, "IMAGE_PULL_BACKOFF": 0.02, "SCHEDULING_ERROR": 0.01,
    })

    set_step(3, {  # "Bump version" keyword → strong dep signal
        "DEPENDENCY_CONFLICT": 0.51, "CONFIG_ERROR": 0.10,
        "INFRA_INCOMPATIBILITY": 0.06, "CODE_REGRESSION": 0.06,
        "ENV_FLAKINESS": 0.05, "TEST_FLAKINESS": 0.04,
        "TOOLING_ARTIFACT": 0.04, "CASCADE_FAILURE": 0.04,
        "NETWORK_TRANSIENT": 0.03, "UNKNOWN": 0.03,
        "OOM_KILL": 0.01, "POD_CRASH": 0.01, "IMAGE_PULL_BACKOFF": 0.01, "SCHEDULING_ERROR": 0.01,
    })

    set_step(4, {  # job_level_fallback → slight ambiguity back, but dep still dominant
        "DEPENDENCY_CONFLICT": 0.47, "CONFIG_ERROR": 0.12,
        "INFRA_INCOMPATIBILITY": 0.09, "CODE_REGRESSION": 0.06,
        "ENV_FLAKINESS": 0.05, "TEST_FLAKINESS": 0.04,
        "TOOLING_ARTIFACT": 0.04, "CASCADE_FAILURE": 0.04,
        "NETWORK_TRANSIENT": 0.03, "UNKNOWN": 0.03,
        "OOM_KILL": 0.01, "POD_CRASH": 0.01, "IMAGE_PULL_BACKOFF": 0.01, "SCHEDULING_ERROR": 0.00,
    })

    set_step(5, {  # inspect_commit_diff: only requirements.txt changed
        "DEPENDENCY_CONFLICT": 0.78, "CONFIG_ERROR": 0.07,
        "INFRA_INCOMPATIBILITY": 0.04, "CODE_REGRESSION": 0.03,
        "ENV_FLAKINESS": 0.02, "TEST_FLAKINESS": 0.02,
        "TOOLING_ARTIFACT": 0.01, "CASCADE_FAILURE": 0.01,
        "NETWORK_TRANSIENT": 0.01, "UNKNOWN": 0.01,
        "OOM_KILL": 0.0, "POD_CRASH": 0.0, "IMAGE_PULL_BACKOFF": 0.0, "SCHEDULING_ERROR": 0.0,
    })

    set_step(6, {  # inspect_dep_changes: exact version pinned + pip error match
        "DEPENDENCY_CONFLICT": 0.94, "CONFIG_ERROR": 0.02,
        "INFRA_INCOMPATIBILITY": 0.01, "CODE_REGRESSION": 0.01,
        "ENV_FLAKINESS": 0.01, "TEST_FLAKINESS": 0.005,
        "TOOLING_ARTIFACT": 0.005, "CASCADE_FAILURE": 0.0,
        "NETWORK_TRANSIENT": 0.0, "UNKNOWN": 0.0,
        "OOM_KILL": 0.0, "POD_CRASH": 0.0, "IMAGE_PULL_BACKOFF": 0.0, "SCHEDULING_ERROR": 0.0,
    })

    # Custom colormap: white → deep blue
    cmap = LinearSegmentedColormap.from_list(
        "belief", ["#F8FAFC", "#BFDBFE", "#2563EB", "#1E3A8A"], N=256
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(history, aspect="auto", cmap=cmap, vmin=0, vmax=0.95, interpolation="nearest")

    # Annotate each cell with the probability value
    for row in range(N_CATS):
        for col in range(n_steps):
            val = history[row, col]
            txt_color = "white" if val > 0.30 else C["dark"]
            weight = "bold" if val > 0.25 else "normal"
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight=weight)

    # Winning category border per step
    for col in range(n_steps):
        winner = int(np.argmax(history[:, col]))
        rect = plt.Rectangle((col - 0.5, winner - 0.5), 1, 1,
                              linewidth=2.5, edgecolor=C["orange"], facecolor="none", zorder=5)
        ax.add_patch(rect)

    # RPA | APA divider
    ax.axvline(rpa_boundary + 0.5, color=C["orange"], linewidth=2.5, linestyle="--", zorder=6)
    ax.text(rpa_boundary + 0.53, -0.85, "← RPA    APA →",
            fontsize=10, color=C["orange"], fontweight="bold", va="top")

    # Axis labels
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(step_labels, fontsize=9.5, rotation=0)
    ax.set_yticks(range(N_CATS))
    ax.set_yticklabels(CATEGORIES_14, fontsize=9.5, fontfamily="monospace")
    ax.set_xlabel("Investigation Step", fontsize=12, labelpad=10)
    ax.set_title(
        "Bayesian Belief State Heatmap: DEPENDENCY_CONFLICT Case\n"
        "Each cell = P(category | evidence so far). Orange border = top category.",
        fontsize=13, fontweight="bold", pad=14
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("Probability", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    path = "images/fig_belief_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[A] Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B – Pipeline Architecture Flow
# Hand-drawn style pipeline: Intake → RPA → Fast-path? → APA loop → Classify
# Uses matplotlib patches so it's fully reproducible (no external assets).
# ─────────────────────────────────────────────────────────────────────────────

def figB_pipeline_flow():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    def box(x, y, w, h, color, label, sublabel=None, text_color="white", fontsize=10):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="white",
            linewidth=2.0, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.13 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22,
                    sublabel, ha="center", va="center",
                    fontsize=8.0, color=text_color, alpha=0.88, style="italic", zorder=4)

    def diamond(cx, cy, hw, hh, color, label, fontsize=9):
        xs = [cx, cx+hw, cx, cx-hw, cx]
        ys = [cy+hh, cy, cy-hh, cy, cy+hh]
        ax.fill(xs, ys, color=color, zorder=3, edgecolor="white", linewidth=2)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=4)

    def arrow(x1, y1, x2, y2, label="", color=C["gray"]):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=2.0, mutation_scale=18), zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.05, my + 0.08, label, fontsize=8.5, color=color,
                    ha="left", va="bottom", style="italic")

    def curved_arrow(x1, y1, x2, y2, color=C["gray"], bend=0.5):
        from matplotlib.patches import FancyArrowPatch
        patch = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={bend}",
            arrowstyle="-|>",
            color=color, linewidth=2.0, mutation_scale=18, zorder=2
        )
        ax.add_patch(patch)

    # ── Column 1: Intake ───────────────────────────────────────────────
    box(0.3, 5.9, 2.6, 1.7, C["teal"], "GHALogs", "CI/CD Failure Event", fontsize=9.5)
    box(0.3, 3.9, 2.6, 1.6, C["teal"], "Intake Parser", "Metadata extraction\nLog windowing", fontsize=9)
    arrow(1.6, 5.9, 1.6, 5.5)
    arrow(1.6, 3.9, 1.6, 3.4)

    # ── Column 2: RPA Bayesian preprocessing ──────────────────────────
    box(0.3, 1.7, 2.6, 1.6, C["orange"], "RPA Pipeline", "7 deterministic signals\nBayesian tracker", fontsize=9)
    arrow(1.6, 3.9, 1.6, 3.3)

    # Fast-path gate (diamond)
    diamond(1.6, 1.0, 1.1, 0.55, C["red"], "conf ≥ 0.75?", fontsize=8)
    arrow(1.6, 1.7, 1.6, 1.55)

    # Fast exit
    ax.annotate("", xy=(3.7, 1.0), xytext=(2.7, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=2.0, mutation_scale=18))
    box(3.7, 0.55, 2.0, 0.90, C["green"], "Early Exit", "Accept RPA result\n(no LLM cost)", fontsize=8.5)
    ax.text(3.05, 1.08, "YES", fontsize=9, color=C["green"], fontweight="bold")

    # No → go to APA
    ax.annotate("", xy=(1.6, 0.45), xytext=(1.6, 0.45),
                arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2.0, mutation_scale=18))
    ax.annotate("", xy=(5.5, 5.0), xytext=(1.6, 0.45),
                arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2.0, mutation_scale=18,
                connectionstyle="arc3,rad=-0.35"))
    ax.text(1.65, 0.38, "NO → hand off to APA", fontsize=8.5, color=C["blue"], fontweight="bold")

    # ── Column 3: APA Loop ────────────────────────────────────────────
    # LangGraph loop background
    loop_rect = mpatches.FancyBboxPatch(
        (5.2, 1.5), 5.5, 5.8,
        boxstyle="round,pad=0.15",
        facecolor="#EFF6FF", edgecolor=C["blue"],
        linewidth=2.5, linestyle="--", zorder=1, alpha=0.85
    )
    ax.add_patch(loop_rect)
    ax.text(7.95, 7.15, "LangGraph Active Investigation Loop",
            ha="center", fontsize=10, color=C["blue"], fontweight="bold", style="italic")

    box(5.5, 5.8, 2.6, 1.1, C["blue"], "Initialize Node", "Seed APA with RPA beliefs\nSet tools_available", fontsize=8.8)
    box(5.5, 4.2, 2.6, 1.2, C["purple"], "Planner Node", "EIG ranking\n+ LLM reasoning", fontsize=8.8)
    box(5.5, 2.4, 2.6, 1.4, C["teal"], "Tool Nodes", "inspect_commit_diff\ncheck_run_history…", fontsize=8.5)

    arrow(6.8, 5.8, 6.8, 5.4)
    arrow(6.8, 4.2, 6.8, 3.8)
    arrow(6.8, 2.4, 6.8, 1.9)

    # Stop criterion diamond
    diamond(6.8, 1.5, 1.0, 0.50, C["red"], "conf ≥ θ\nor K_max?", fontsize=7.5)

    # Loop back arrow
    curved_arrow(7.8, 1.5, 8.3, 4.8, color=C["purple"], bend=-0.35)
    ax.text(8.35, 3.4, "NO: re-plan", fontsize=8.5, color=C["purple"], style="italic", rotation=-90)

    # Tool → Planner feedback
    curved_arrow(5.5, 3.0, 5.3, 4.5, color=C["teal"], bend=0.5)
    ax.text(4.6, 3.7, "Bayesian\nupdate", fontsize=8, color=C["teal"], ha="center")

    # YES → Classify
    ax.annotate("", xy=(11.2, 1.5), xytext=(7.8, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=2.0, mutation_scale=18))
    ax.text(9.2, 1.58, "YES", fontsize=9, color=C["green"], fontweight="bold")

    # ── Column 4: Classify ────────────────────────────────────────────
    box(11.2, 0.8, 2.5, 1.4, C["green"], "Classify Node", "Synthesise evidence\nFinal JSON output", fontsize=9)

    # Shared Bayesian tracker label
    tracker_rect = mpatches.FancyBboxPatch(
        (0.0, 0.0), 3.2, 0.36,
        boxstyle="round,pad=0.06",
        facecolor=C["lgray"], edgecolor=C["gray"],
        linewidth=1.5, zorder=1
    )
    ax.add_patch(tracker_rect)
    ax.text(1.6, 0.18, "Shared: Bayesian Belief Tracker  (14 categories, Shannon entropy)",
            ha="center", va="center", fontsize=8, color=C["dark"])

    ax.set_title(
        "APA Framework: End-to-End Diagnostic Pipeline",
        fontsize=14, fontweight="bold", pad=10, color=C["dark"]
    )

    plt.tight_layout()
    path = "images/fig_pipeline_flow.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[B] Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure C – Entropy Convergence Funnel
# Shows three representative case trajectories converging from high entropy
# to below the stop threshold. Zones are shaded:
#   Red zone  = needs investigation
#   Yellow    = getting there
#   Green     = classified
# This is the most visually compelling figure for explaining why the APA loop stops.
# ─────────────────────────────────────────────────────────────────────────────

def figC_entropy_funnel():
    MAX_H = math.log2(N_CATS)     # 3.807 bits
    STOP_H = 1.0                  # θ_stop
    FAST_CONF = 0.75
    FAST_H = (1.0 - FAST_CONF) * MAX_H  # ≈ 0.952 bits

    # Three representative trajectories (different cases, different convergence speeds)
    steps_max = 8
    traj = {
        "Quick convergence\n(DEPENDENCY_CONFLICT)": {
            "color": C["blue"],
            "lw": 2.8,
            "ls": "-",
            "h": [3.78, 2.95, 2.10, 1.45, 0.82],  # converges at step 4
            "marker": "o",
        },
        "Moderate case\n(CONFIG_ERROR)": {
            "color": C["orange"],
            "lw": 2.5,
            "ls": "-",
            "h": [3.80, 3.40, 3.05, 2.60, 2.10, 1.55, 0.92],  # converges at step 6
            "marker": "s",
        },
        "Hard case — hits K_max\n(ambiguous: CODE_REG vs DEPENDENCY)": {
            "color": C["red"],
            "lw": 2.5,
            "ls": "--",
            "h": [3.82, 3.65, 3.30, 3.00, 2.78, 2.55, 2.30, 2.10],  # never drops below STOP_H
            "marker": "^",
        },
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Shaded zones
    ax.axhspan(STOP_H, MAX_H + 0.05,  alpha=0.08, color=C["red"],    zorder=0)
    ax.axhspan(0,       STOP_H,        alpha=0.10, color=C["green"],  zorder=0)

    ax.text(7.6, (STOP_H + MAX_H)/2, "Investigation\nzone",
            fontsize=9.5, color=C["red"], ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6))
    ax.text(7.6, STOP_H/2, "Classified",
            fontsize=9.5, color=C["green"], ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6))

    # Threshold lines
    ax.axhline(STOP_H, color=C["green"], linewidth=2.0, linestyle="--", zorder=2,
               label=f"Stop threshold θ_stop = {STOP_H:.1f} bit")
    ax.axhline(MAX_H, color=C["gray"], linewidth=1.5, linestyle=":", zorder=2, alpha=0.7,
               label=f"H_max = {MAX_H:.2f} bits (uniform, 14 cats)")

    # Trajectories
    for name, t in traj.items():
        xs = list(range(len(t["h"])))
        ax.plot(xs, t["h"], label=name, color=t["color"],
                linewidth=t["lw"], linestyle=t["ls"],
                marker=t["marker"], markersize=8, zorder=3)
        # Mark the stop point (first time below threshold)
        for i, h in enumerate(t["h"]):
            if h < STOP_H:
                ax.plot(i, h, "*", color=t["color"], markersize=18, zorder=5)
                ax.annotate(f"  Classified\n  step {i}",
                            xy=(i, h), xytext=(i + 0.25, h + 0.35),
                            fontsize=8.5, color=t["color"],
                            arrowprops=dict(arrowstyle="-", color=t["color"], lw=1.2))
                break

    # K_max marker for the hard case
    ax.axvline(7, color=C["red"], linewidth=1.5, linestyle=":", alpha=0.7, zorder=2)
    ax.text(7.05, 3.5, "K_max = 7", fontsize=9, color=C["red"], style="italic")

    ax.set_xlabel("Investigation Step", fontsize=12, labelpad=8)
    ax.set_ylabel("Shannon Entropy H(P)  (bits)", fontsize=12, labelpad=8)
    ax.set_xlim(-0.3, 7.8)
    ax.set_ylim(-0.15, MAX_H + 0.25)
    ax.set_xticks(range(8))
    ax.set_xticklabels(["Init"] + [f"Step {i}" for i in range(1, 8)], fontsize=10)

    # Secondary y-axis: normalised confidence
    ax2 = ax.twinx()
    ax2.set_ylim(-0.15, MAX_H + 0.25)
    conf_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
    h_ticks = [(1 - c) * MAX_H for c in conf_ticks]
    ax2.set_yticks(h_ticks)
    ax2.set_yticklabels([f"{c:.0%}" for c in conf_ticks], fontsize=10)
    ax2.set_ylabel("Normalised Confidence  (1 − H/H_max)", fontsize=11)

    ax.set_title(
        "Entropy Convergence Trajectories: Three Representative CI Failure Cases\n"
        "Agent loop terminates when H(P) < θ_stop = 1.0 bit (★) or at K_max = 7 steps",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.22, linestyle="--")

    plt.tight_layout()
    path = "images/fig_entropy_funnel.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[C] Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure D – Tool Capability Radar
# For each tool, shows its diagnostic coverage across 6 failure-type groups.
# A radar (spider) chart — immediately shows which tools are specialists vs generalists.
# ─────────────────────────────────────────────────────────────────────────────

def figD_tool_radar():
    # Failure-type axes for radar (condensed from 14 → 6 groups)
    axes = [
        "Code/Regression",
        "Dependencies",
        "Config/Workflow",
        "Infra/Runner",
        "Flakiness",
        "Kubernetes/OOM",
    ]

    # Tool scores on each axis (0–1), derived from OUTCOME_SCENARIOS domain knowledge
    # Higher = tool can produce strong signal for this failure group
    tools_data = {
        "deep_log_analysis":          [0.80, 0.70, 0.50, 0.60, 0.65, 0.40],
        "inspect_commit_diff":        [0.75, 0.65, 0.70, 0.35, 0.25, 0.10],
        "check_run_history":          [0.60, 0.40, 0.30, 0.45, 0.90, 0.20],
        "inspect_dependency_changes": [0.30, 0.95, 0.20, 0.40, 0.10, 0.05],
        "inspect_workflow_file":      [0.10, 0.20, 0.90, 0.55, 0.15, 0.05],
        "inspect_runner_environment": [0.15, 0.50, 0.40, 0.90, 0.30, 0.35],
        "inspect_k8s_events":         [0.20, 0.10, 0.25, 0.40, 0.15, 0.95],
    }

    tool_colors = [C["blue"], C["orange"], C["green"], C["purple"],
                   C["teal"], C["red"], C["yellow"]]

    N = len(axes)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    # Grid
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_tick_params(labelleft=False)
    for r in [0.25, 0.50, 0.75, 1.0]:
        ax.plot(angles, [r] * (N+1), color=C["lgray"], linewidth=0.8, zorder=0)
        ax.text(0.05, r, f"{r:.0%}", ha="left", va="center",
                fontsize=8, color=C["gray"], transform=ax.get_yaxis_transform())

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=11, fontweight="bold")

    for (tool, scores), color in zip(tools_data.items(), tool_colors):
        vals = scores + scores[:1]
        ax.plot(angles, vals, linewidth=2.0, linestyle="-", color=color,
                label=tool, zorder=2)
        ax.fill(angles, vals, color=color, alpha=0.06, zorder=1)

    ax.set_title(
        "Diagnostic Tool Capability Coverage\nStrength across 6 failure-type groups",
        fontsize=13, fontweight="bold", pad=22
    )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.17),
        ncol=2, fontsize=9.5, framealpha=0.9
    )

    plt.tight_layout()
    path = "images/fig_tool_radar.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[D] Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure E – RPA 7-Signal Information Gain Waterfall
# Horizontal waterfall: each signal is a bar showing how much entropy it removed.
# Remaining entropy is shown as a thin grey extension.
# This is the clearest possible explanation of what the 7 RPA signals do.
# ─────────────────────────────────────────────────────────────────────────────

def figE_signal_cascade():
    MAX_H = math.log2(N_CATS)

    # Entropy after each signal, for three case archetypes
    cases = {
        "Clear DEPENDENCY_CONFLICT case\n(dependabot bump, requirements.txt)": {
            "color": C["blue"],
            "entropy": [MAX_H, 3.15, 3.10, 2.12, 2.05, 1.60, 1.55, 1.35],
            "signals": ["Init", "branch_type", "jobs_failed", "commit_msg",
                        "detect_mode", "error_text", "file_paths", "corpus_sim"],
        },
        "Ambiguous case\n(CODE_REG vs TEST_FLAKINESS)": {
            "color": C["orange"],
            "entropy": [MAX_H, 3.20, 3.05, 3.00, 2.95, 2.75, 2.70, 2.60],
            "signals": ["Init", "branch_type", "jobs_failed", "commit_msg",
                        "detect_mode", "error_text", "file_paths", "corpus_sim"],
        },
        "All-jobs-fail CONFIG case\n(YAML syntax error)": {
            "color": C["green"],
            "entropy": [MAX_H, 3.50, 2.40, 2.38, 1.80, 1.20, 1.10, 0.90],
            "signals": ["Init", "branch_type", "jobs_failed", "commit_msg",
                        "detect_mode", "error_text", "file_paths", "corpus_sim"],
        },
    }

    signal_labels = ["Init\n(uniform)", "$L_1$\nbranch_type", "$L_2$\njobs_failed",
                     "$L_3$\ncommit_msg", "$L_4$\ndetect_mode", "$L_5$\nerror_text",
                     "$L_6$\nfile_paths", "$L_7$\ncorpus_sim"]

    fig, (ax_main, ax_gain) = plt.subplots(
        2, 1, figsize=(13, 8.5), gridspec_kw={"height_ratios": [3, 1.6]}
    )

    # ── Top: entropy over steps (line + shading) ──────────────────────
    for name, case in cases.items():
        xs = list(range(len(case["entropy"])))
        ax_main.plot(xs, case["entropy"], color=case["color"],
                     linewidth=2.8, marker="o", markersize=8, label=name, zorder=3)
        ax_main.fill_between(xs, case["entropy"], 0, color=case["color"], alpha=0.07, zorder=1)

    ax_main.axhline(1.0, color=C["green"], linewidth=2.0, linestyle="--",
                    label="APA stop threshold (θ = 1.0 bit)", zorder=2)
    ax_main.axhline(MAX_H, color=C["gray"], linewidth=1.2, linestyle=":",
                    alpha=0.7, label=f"H_max = {MAX_H:.2f} bits", zorder=2)

    # Shade "APA territory" — below θ_stop
    ax_main.axhspan(0, 1.0, alpha=0.06, color=C["green"], zorder=0)
    ax_main.text(7.35, 0.50, "Classified\n(below θ_stop)", fontsize=8.5,
                 color=C["green"], ha="right", va="center")
    ax_main.text(7.35, 2.40, "Needs APA\nagent loop", fontsize=8.5,
                 color=C["orange"], ha="right", va="center")

    ax_main.set_xlim(-0.3, 7.45)
    ax_main.set_ylim(0, MAX_H + 0.3)
    ax_main.set_xticks(range(8))
    ax_main.set_xticklabels(signal_labels, fontsize=9.5)
    ax_main.set_ylabel("Shannon Entropy H(P)  (bits)", fontsize=11)
    ax_main.set_title(
        "RPA 7-Signal Entropy Reduction: Three Case Archetypes\n"
        "Shows which signals are most informative per case type",
        fontsize=13, fontweight="bold", pad=12
    )
    ax_main.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    ax_main.grid(True, alpha=0.22, linestyle="--")

    # ── Bottom: per-signal information gain bars ──────────────────────
    n_sigs = 7  # signals L1..L7
    x = np.arange(n_sigs)
    w = 0.25

    for i, (name, case) in enumerate(cases.items()):
        ig = [case["entropy"][s] - case["entropy"][s+1] for s in range(n_sigs)]
        ig = [max(0, v) for v in ig]
        offset = (i - 1) * w
        ax_gain.bar(x + offset, ig, width=w, color=case["color"],
                    alpha=0.80, edgecolor="white", linewidth=0.8, zorder=2)

    sig_xlabels = ["$L_1$\nbranch", "$L_2$\njobs", "$L_3$\nmsg",
                   "$L_4$\ndetect", "$L_5$\nerror", "$L_6$\nfiles", "$L_7$\ncorpus"]
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels(sig_xlabels, fontsize=10)
    ax_gain.set_ylabel("Information Gain\n(bits)", fontsize=10)
    ax_gain.set_title("Per-Signal Information Gain", fontsize=11, fontweight="bold", pad=6)
    ax_gain.grid(axis="y", alpha=0.25, linestyle="--")
    ax_gain.set_xlim(-0.5, 6.5)

    plt.tight_layout(h_pad=1.2)
    path = "images/fig_signal_cascade.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[E] Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating creative methodology figures...\n")
    figA_belief_heatmap()
    figB_pipeline_flow()
    figC_entropy_funnel()
    figD_tool_radar()
    figE_signal_cascade()
    print("\nDone. All figures saved to images/")
