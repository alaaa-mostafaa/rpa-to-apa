# eval_decision_layer.py
# ─────────────────────────────────────────────────────────────────────
# Evaluate the decision layer by running existing classification
# results through the triage/trust/policy logic.
#
# Compares RPA-only vs APA action recommendations to show that
# APA's higher confidence leads to more autonomous (T2) decisions.
#
# Usage:
#   python eval_decision_layer.py --results honest_eval_results.json
#   python eval_decision_layer.py --results test_agent_batch_results.json
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.apa.decision_layer import (
    recommend_action,
    assign_trust_tier,
    recommend_canary_action,
    recommend_flag_action,
    enrich_result,
)


def _load_results(path: Path) -> List[Dict[str, Any]]:
    """Load evaluation results from JSON (list or object with 'results' key)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "cases", "evaluations"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Cannot find results list in {path}")


def _extract_classification(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract classification dict from various result formats."""
    if "classification" in result:
        return result["classification"]
    # Flat format (some eval outputs have category/confidence at top level)
    if "category" in result:
        return {
            "category": result["category"],
            "confidence": result.get("confidence", 0.0),
            "severity": result.get("severity", "MODERATE"),
        }
    return {}


def analyze_decisions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run all results through the decision layer and compute stats."""
    action_counts = Counter()
    tier_counts = Counter()
    canary_counts = Counter()
    flag_counts = Counter()
    category_action_matrix: Dict[str, Counter] = {}

    # RPA vs APA comparison (if results have both)
    rpa_actions = Counter()
    apa_actions = Counter()
    rpa_tiers = Counter()
    apa_tiers = Counter()
    rpa_apa_agree = 0
    rpa_apa_total = 0

    enriched_results = []

    for r in results:
        cl = _extract_classification(r)
        if not cl or not cl.get("category"):
            continue

        category = cl["category"]
        confidence = float(cl.get("confidence", 0.0))
        severity = cl.get("severity", "MODERATE")

        action = recommend_action(category, confidence)
        tier = assign_trust_tier(confidence, category)
        canary = recommend_canary_action(category, confidence)
        flag = recommend_flag_action(category, confidence)

        action_counts[action] += 1
        tier_counts[tier] += 1
        canary_counts[canary] += 1
        flag_counts[flag] += 1

        if category not in category_action_matrix:
            category_action_matrix[category] = Counter()
        category_action_matrix[category][action] += 1

        # Track if this is APA or RPA based on fast_path / tools_used
        is_fast_path = r.get("fast_path", False)
        tools_used = r.get("tools_used", [])
        agent_mode = "RPA" if (is_fast_path or not tools_used) else "APA"

        enriched = {
            "category": category,
            "confidence": confidence,
            "action": action,
            "trust_tier": tier,
            "canary_action": canary,
            "flag_action": flag,
            "agent_mode": agent_mode,
        }
        enriched_results.append(enriched)

        if agent_mode == "APA":
            apa_actions[action] += 1
            apa_tiers[tier] += 1
        else:
            rpa_actions[action] += 1
            rpa_tiers[tier] += 1

        # Simulate RPA decision (using Bayesian confidence only)
        beliefs = r.get("beliefs", {})
        if beliefs:
            bayes_top = max(beliefs, key=beliefs.get)
            bayes_conf_raw = beliefs[bayes_top]
            # Bayesian confidence = 1 - H/H_max (approximate from top probability)
            rpa_action = recommend_action(bayes_top, bayes_conf_raw)
            rpa_tier = assign_trust_tier(bayes_conf_raw, bayes_top)

            rpa_apa_total += 1
            if rpa_action == action:
                rpa_apa_agree += 1

    return {
        "n_cases": len(enriched_results),
        "action_distribution": dict(action_counts.most_common()),
        "tier_distribution": dict(tier_counts.most_common()),
        "canary_distribution": dict(canary_counts.most_common()),
        "flag_distribution": dict(flag_counts.most_common()),
        "category_action_matrix": {
            cat: dict(counts.most_common()) for cat, counts in sorted(category_action_matrix.items())
        },
        "rpa_vs_apa": {
            "apa_action_distribution": dict(apa_actions.most_common()),
            "rpa_action_distribution": dict(rpa_actions.most_common()),
            "apa_tier_distribution": dict(apa_tiers.most_common()),
            "rpa_tier_distribution": dict(rpa_tiers.most_common()),
            "action_agreement_rate": (rpa_apa_agree / rpa_apa_total) if rpa_apa_total else None,
            "n_compared": rpa_apa_total,
        },
        "enriched_results": enriched_results,
    }


def print_summary(stats: Dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print(f"\n{'=' * 70}")
    print("DECISION LAYER EVALUATION")
    print(f"{'=' * 70}")
    print(f"  Cases analyzed: {stats['n_cases']}")

    print(f"\n  --- Action Distribution ---")
    for action, count in stats["action_distribution"].items():
        pct = count / stats["n_cases"] * 100
        bar = "#" * int(pct / 2)
        print(f"    {action:<20} {count:>5}  ({pct:5.1f}%)  {bar}")

    print(f"\n  --- Trust Tier Distribution ---")
    for tier, count in stats["tier_distribution"].items():
        pct = count / stats["n_cases"] * 100
        labels = {"T0": "observe", "T1": "recommend", "T2": "autonomous"}
        print(f"    {tier} ({labels.get(tier, '?'):<12}) {count:>5}  ({pct:5.1f}%)")

    print(f"\n  --- Canary Decision Distribution ---")
    for action, count in stats["canary_distribution"].items():
        pct = count / stats["n_cases"] * 100
        print(f"    {action:<20} {count:>5}  ({pct:5.1f}%)")

    print(f"\n  --- Category x Action Matrix ---")
    matrix = stats["category_action_matrix"]
    all_actions = sorted(set(a for counts in matrix.values() for a in counts))
    header = f"    {'category':<25}" + "".join(f" {a:<14}" for a in all_actions)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for cat in sorted(matrix):
        row = f"    {cat:<25}"
        for a in all_actions:
            count = matrix[cat].get(a, 0)
            row += f" {count:<14}"
        print(row)

    rva = stats["rpa_vs_apa"]
    if rva.get("n_compared"):
        print(f"\n  --- RPA vs APA Action Agreement ---")
        print(f"    Agreement rate: {rva['action_agreement_rate']:.1%} ({rva['n_compared']} cases)")

    print(f"\n{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate decision layer on existing results")
    parser.add_argument("--results", type=Path, required=True, help="Path to evaluation results JSON")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: {args.results} not found", file=sys.stderr)
        sys.exit(1)

    results = _load_results(args.results)
    print(f"Loaded {len(results)} results from {args.results}")

    stats = analyze_decisions(results)
    print_summary(stats)

    if args.out:
        # Don't save the full enriched_results in the summary (too large)
        save_stats = {k: v for k, v in stats.items() if k != "enriched_results"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(save_stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Saved to {args.out}")


if __name__ == "__main__":
    main()
