#!/usr/bin/env python
# evals/run_apa_retrieval.py
# ─────────────────────────────────────────────────────────────────────
# L1 / L2 / L3 comparison on the existing benchmark_final_eval.json.
#
# L1 = RPA  (already scored, read from benchmark file)
# L2 = APA  (already scored, read from benchmark file)
# L3 = APA+Retrieval — recompute what the APA prior WOULD have been
#      if the retrieval store had been used, then re-judge whether the
#      prior shift would have changed the outcome.
#
# Because we don't re-run the full LangGraph agent (expensive), we
# measure the retrieval contribution directly:
#   - Does the L3 prior rank the correct category higher than L2's
#     uniform prior, on cases where L2 was WRONG or PARTIAL?
#   - We call this a "retrieval rescue" — the prior alone would have
#     changed the agent's starting belief in the right direction.
#
# Leave-one-out: each case is temporarily removed from the store
# before querying so we never retrieve the exact same run_id.
#
# Output:
#   data/l3_retrieval_eval.json   — per-case results
#   Prints L1 / L2 / L3 accuracy table to stdout
# ─────────────────────────────────────────────────────────────────────

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.apa.chroma_case_store import ChromaCaseStore

BENCHMARK_PATH = _ROOT / "data" / "benchmark_final_eval.json"
OUTPUT_PATH    = _ROOT / "data" / "l3_retrieval_eval.json"
CHROMA_PATH    = str(_ROOT / "data" / "chroma")


def prior_top_category(prior: dict) -> str:
    return max(prior, key=prior.get)


def prior_rank_of(prior: dict, category: str) -> int:
    """1-indexed rank of category in prior (1 = highest probability)."""
    ranked = sorted(prior, key=prior.get, reverse=True)
    try:
        return ranked.index(category) + 1
    except ValueError:
        return len(prior)


# Map ground-truth developer actions → expected failure categories
GT_TO_CATEGORY = {
    "CODE_FIX":           "CODE_REGRESSION",
    "CODE_CHANGE":        "CODE_REGRESSION",
    "REVERT":             "CODE_REGRESSION",
    "WORKFLOW_FIX":       "CONFIG_ERROR",
    "PIN_VERSION":        "DEPENDENCY_CONFLICT",
    "PR_MERGED":          "CODE_REGRESSION",
    "PR_MERGED_UNCLEAR":  "CODE_REGRESSION",
    "RETRY":              "ENV_FLAKINESS",
}


def main():
    print(f"Loading benchmark: {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH) as f:
        data = json.load(f)

    scorable = [
        d for d in data
        if d["rpa"]["judge"]["verdict"] != "NOT_SCORABLE"
        and d["apa_eig"].get("prediction")
    ]
    print(f"Scorable cases: {len(scorable)} / {len(data)}")

    store = ChromaCaseStore(path=CHROMA_PATH)
    print(f"Chroma store: {store.count()} cases\n")

    results = []
    retrieval_rescues   = 0   # L2 wrong/partial → L3 prior points right
    retrieval_no_change = 0   # L3 prior agrees with L2 outcome
    retrieval_hurts     = 0   # L3 prior points wrong on an L2-correct case

    for i, rec in enumerate(scorable):
        run_id    = rec["run_id"]
        commit    = rec.get("commit", "")
        repo      = rec.get("repo", "")
        gt_action = rec["ground_truth"]["action"]
        gt_cat    = GT_TO_CATEGORY.get(gt_action, "")

        rpa_verdict = rec["rpa"]["judge"]["verdict"]
        apa_verdict = rec["apa_eig"]["judge"]["verdict"]
        apa_cat     = rec["apa_eig"]["prediction"].get("category", "")

        # Temporarily remove this case from the store (leave-one-out)
        was_in_store = False
        try:
            existing = store._collection.get(ids=[run_id], include=[])
            if existing["ids"]:
                store._collection.delete(ids=[run_id])
                was_in_store = True
        except Exception:
            pass

        # Compute retrieval prior (blends with uniform as independent signal)
        prior, neighbours = store.compute_prior(
            commit_title=commit,
            error_lines=rec.get("error_lines", []),
            verbose=False,
        )

        # Restore the case
        if was_in_store:
            store.upsert_case(
                run_id=run_id,
                commit_title=commit,
                error_lines=rec.get("error_lines", []),
                category=apa_cat,
                gt_verdict=apa_verdict,
                repo=repo,
            )

        prior_top  = prior_top_category(prior)
        prior_rank = prior_rank_of(prior, gt_cat) if gt_cat else None
        uniform_rank = prior_rank_of(
            {c: 1.0/14 for c in prior}, gt_cat
        ) if gt_cat else None

        # L3 verdict logic:
        #
        # RESCUE: L2 was WRONG/PARTIAL AND retrieval prior ranks the
        #   correct category at rank 1 (prior would have nudged agent right).
        #
        # HURT: L2 was CORRECT AND the prior actively buries the correct
        #   category (rank >= 5 out of 14) — it would mislead the agent.
        #   We do NOT count it as HURT when prior_top != gt but gt is still
        #   rank 2-4 (prior is slightly off but not actively harmful).
        #
        # SAME: everything else — prior either agrees or is neutral.
        l3_prior_rank1 = (
            gt_cat and prior_rank is not None and prior_rank == 1
        )
        l3_prior_buries = (
            gt_cat and prior_rank is not None and prior_rank >= 5
        )

        if apa_verdict in ("WRONG", "PARTIAL") and l3_prior_rank1:
            retrieval_rescues += 1
            l3_outcome = "RESCUE"
        elif apa_verdict == "CORRECT" and l3_prior_buries:
            retrieval_hurts += 1
            l3_outcome = "HURT"
        else:
            retrieval_no_change += 1
            l3_outcome = "SAME"

        results.append({
            "run_id":         run_id,
            "repo":           repo,
            "commit":         commit,
            "gt_action":      gt_action,
            "gt_category":    gt_cat,
            "rpa_verdict":    rpa_verdict,
            "apa_verdict":    apa_verdict,
            "apa_category":   apa_cat,
            "prior_top":      prior_top,
            "prior_rank_gt":  prior_rank,
            "n_neighbours":   len(neighbours),
            "neighbours":     neighbours[:3],
            "l3_outcome":     l3_outcome,
        })

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(scorable)} processed...")

    # ── summary table ─────────────────────────────────────────────────
    total = len(scorable)
    rpa_correct  = sum(1 for r in results if r["rpa_verdict"] == "CORRECT")
    rpa_partial  = sum(1 for r in results if r["rpa_verdict"] == "PARTIAL")
    apa_correct  = sum(1 for r in results if r["apa_verdict"] == "CORRECT")
    apa_partial  = sum(1 for r in results if r["apa_verdict"] == "PARTIAL")

    # L3 strict = APA correct + WRONG cases where prior top = correct category
    # L3 partial = L2 partial + WRONG cases rescued (rank<=2) - hurt cases
    # Do NOT add PARTIAL rescues to partial estimate — they're already in apa_partial.
    strict_rescues = sum(
        1 for r in results
        if r["l3_outcome"] == "RESCUE" and r["apa_verdict"] == "WRONG"
    )
    wrong_rank2_rescues = sum(
        1 for r in results
        if r["apa_verdict"] == "WRONG" and r.get("prior_rank_gt") == 2
    )
    l3_strict_estimate  = apa_correct + strict_rescues
    l3_partial_estimate = apa_correct + apa_partial + strict_rescues + wrong_rank2_rescues - retrieval_hurts

    print()
    print("=" * 62)
    print(f"{'':4} {'STRICT':>10} {'STRICT%':>10} {'±PARTIAL':>10} {'±PARTIAL%':>10}")
    print("-" * 62)
    print(f"{'L1 RPA':<12} {rpa_correct:>10} {rpa_correct/total:>10.1%} "
          f"{rpa_correct+rpa_partial:>10} {(rpa_correct+rpa_partial)/total:>10.1%}")
    print(f"{'L2 APA':<12} {apa_correct:>10} {apa_correct/total:>10.1%} "
          f"{apa_correct+apa_partial:>10} {(apa_correct+apa_partial)/total:>10.1%}")
    print(f"{'L3 APA+R':<12} {'~'+str(l3_strict_estimate):>10} "
          f"{'~'+f'{l3_strict_estimate/total:.1%}':>10} "
          f"{'~'+str(l3_partial_estimate):>10} "
          f"{'~'+f'{l3_partial_estimate/total:.1%}':>10}")
    print("=" * 62)
    print(f"\nRetrieval outcomes on {total} scorable cases:")
    print(f"  RESCUE  (prior rank-1 = gt, apa was wrong/partial): {retrieval_rescues}")
    print(f"    of which: apa=WRONG rescued to strict: {strict_rescues}")
    print(f"    of which: apa=PARTIAL rescued (already in partial): {retrieval_rescues - strict_rescues}")
    print(f"  HURT    (prior buries gt rank>=5 on correct case):   {retrieval_hurts}")
    print(f"  SAME    (neutral or minor shift):                    {retrieval_no_change}")
    print(f"  WRONG->rank2 (soft improvement, partial only):       {wrong_rank2_rescues}")

    # Prior rank distribution
    rank_dist = Counter(r["prior_rank_gt"] for r in results if r["prior_rank_gt"])
    print(f"\nPrior rank of correct category:")
    for rank in sorted(rank_dist):
        print(f"  rank {rank}: {rank_dist[rank]} cases ({rank_dist[rank]/total:.1%})")

    # Cases with 0 neighbours
    no_neighbours = sum(1 for r in results if r["n_neighbours"] == 0)
    print(f"\nCases with no neighbours above threshold: {no_neighbours}/{total}")

    # ── save results ──────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_scorable": total,
                "l1_rpa_strict":   rpa_correct,
                "l1_rpa_partial":  rpa_correct + rpa_partial,
                "l2_apa_strict":   apa_correct,
                "l2_apa_partial":  apa_correct + apa_partial,
                "l3_strict_estimate":  l3_strict_estimate,
                "l3_partial_estimate": l3_partial_estimate,
                "retrieval_rescues":   retrieval_rescues,
                "retrieval_hurts":     retrieval_hurts,
                "retrieval_no_change": retrieval_no_change,
                "rank1_rescues":       strict_rescues,
            },
            "cases": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
