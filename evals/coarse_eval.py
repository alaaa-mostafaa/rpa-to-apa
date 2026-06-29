#!/usr/bin/env python3
"""
coarse_eval.py
==============
Score RPA-vs-APA at the resolution the GROUND TRUTH can actually confirm.

Motivation
----------
The 9 prediction categories are FINER than file-based ground truth can resolve.
A developer's changed files can only distinguish a few "where did it break"
buckets — a lint fix and a logic-bug fix both just "edit source code", so
QUALITY_VIOLATION can never be confirmed as distinct from CODE_REGRESSION by
files alone (see evals/evaluation_judge.py:158-182). Scoring the fine categories
against file GT therefore PUNISHES the finer call and is invalid.

This scorer collapses BOTH the prediction and the developer action into coarse
buckets the file GT *can* confirm, then a prediction is CORRECT iff its bucket
matches the developer-fix bucket. This:
  * lets every fine category (QUALITY_VIOLATION, INFRA_INCOMPATIBILITY, ...) be
    CORRECT when it lands in the right coarse bucket,
  * is fully DETERMINISTIC from the stored (predicted_category, gt_action) — no
    LLM judge, so it is immune to the judge-version contamination in old result
    files and is perfectly reproducible,
  * reports MACRO accuracy (mean of per-bucket accuracy) so an imbalanced corpus
    and a constant majority-class baseline (RPA always predicts CODE_REGRESSION)
    cannot inflate the headline number.

Usage:
  python evals/coarse_eval.py data/eval_500_results.jsonl [more.jsonl ...]
  python evals/coarse_eval.py --balanced data/eval_500_results.jsonl  # equal-n subsample
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

# ── Coarse taxonomy ───────────────────────────────────────────────────────
# Four buckets, each defined by the developer-fix signal that confirms it.
# CODE       : the defect was in application/test source -> dev edits source/test
# DEPENDENCY : a dependency/version problem            -> dev pins/bumps a manifest
# CONFIG     : CI / workflow / runner configuration    -> dev edits a workflow file
# TRANSIENT  : flaky / environmental, no code fix      -> dev just re-runs (retry)

PREDICTION_TO_COARSE = {
    "CODE_REGRESSION":     "CODE",
    "QUALITY_VIOLATION":   "CODE",        # lint/style/structure = a source edit
    "TEST_FLAKINESS":      "CODE",        # flaky test is usually fixed by editing the test
    "DEPENDENCY_CONFLICT": "DEPENDENCY",
    "CONFIG_ERROR":        "CONFIG",
    "INFRA_INCOMPATIBILITY": "CONFIG",    # runner/toolchain mismatch = CI config
    "ENV_FLAKINESS":       "TRANSIENT",
    "CASCADE_FAILURE":     "TRANSIENT",
    "TOOLING_ARTIFACT":    "TRANSIENT",
}

GT_ACTION_TO_COARSE = {
    "CODE_FIX":          "CODE",
    "CODE_CHANGE":       "CODE",
    "PIN_VERSION":       "DEPENDENCY",
    "DEPENDENCY_CHANGE": "DEPENDENCY",
    "WORKFLOW_FIX":      "CONFIG",
    "RETRY_SUCCEEDED":   "TRANSIENT",
    # Everything else is NOT_SCORABLE (no usable, unambiguous developer fix):
    #   REVERT (doesn't pin a category), PR_MERGED_* / NO_* / UNKNOWN / AMBIGUOUS.
}

BUCKETS = ["CODE", "DEPENDENCY", "CONFIG", "TRANSIENT"]


def pred_bucket(category) -> str | None:
    return PREDICTION_TO_COARSE.get((category or "").upper())


def gt_bucket(dev_action) -> str | None:
    return GT_ACTION_TO_COARSE.get((dev_action or "").upper())


def coarse_verdict(category, dev_action) -> str:
    """CORRECT / WRONG / NOT_SCORABLE from stored prediction + developer action."""
    gtb = gt_bucket(dev_action)
    if gtb is None:
        return "NOT_SCORABLE"
    pb = pred_bucket(category)
    if pb is None:
        return "WRONG"          # made a prediction outside the scorable space
    return "CORRECT" if pb == gtb else "WRONG"


# ── Substantive-fix re-bucketing (ground-truth quality correction) ──────────
# The developer-action label over-tags DEPENDENCY whenever a lockfile auto-updates
# (package-lock.json / go.sum / yarn.lock) alongside a source or workflow fix. A
# 12-case audit (2026-06-15) found ~6/12 DEPENDENCY-labeled cases were really
# CODE/CONFIG/docs by this measure. To score against what the developer
# SUBSTANTIVELY changed, re-derive the fix's bucket(s) from the changed files in
# the GT reasoning, IGNORING lockfile noise, with priority CODE > CONFIG >
# DEPENDENCY (a source/workflow edit is a deliberate fix; a lockfile bump is
# usually incidental). Docs/version-only fixes return None -> NOT_SCORABLE.
import re as _re
_FIX_WORKFLOW_RE = _re.compile(r"\.github/|/workflows/|\.circleci|dockerfile\b|action\.ya?ml|\.pre-commit-config|tox\.ini|setup\.cfg|\.flake8|\.eslintrc|\.prettierrc|\.editorconfig|\.golangci|pytest\.ini|noxfile|\.yamllint|codecov|\.readthedocs|mkdocs\.ya?ml", _re.I)
_FIX_MANIFEST_RE = _re.compile(r"package\.json|go\.mod\b|requirements[\w.-]*\.txt|cargo\.toml|pyproject\.toml|\bgemfile\b|pom\.xml|build\.gradle", _re.I)
_FIX_SOURCE_RE   = _re.compile(r"\.(js|jsx|ts|tsx|py|java|kt|go|rs|cpp|cc|rb|php|cs|swift|scala|pyx|pxd)\b", _re.I)
_FIX_LOCK_RE     = _re.compile(r"package-lock|yarn\.lock|pnpm-lock|go\.sum|cargo\.lock|poetry\.lock", _re.I)


def substantive_fix_buckets(reasoning):
    """(primary_bucket, {all_buckets}) the developer SUBSTANTIVELY changed.

    Returns (None, None) for docs/version-only or unparseable fixes (NOT_SCORABLE).
    """
    t = reasoning or ""
    b = set()
    if _FIX_SOURCE_RE.search(t):   b.add("CODE")
    if _FIX_WORKFLOW_RE.search(t): b.add("CONFIG")
    if _FIX_MANIFEST_RE.search(t): b.add("DEPENDENCY")
    if not b and _FIX_LOCK_RE.search(t): b.add("DEPENDENCY")
    if not b:
        return None, None
    for p in ("CODE", "CONFIG", "DEPENDENCY"):
        if p in b:
            return p, b


def substantive_credit(category, reasoning, partial: bool = True, multilabel: bool = False):
    """Credit for one prediction against the developer's (multi-category) fix.

    - 1.0 if it hits the PRIMARY substantive bucket.
    - For a bucket the fix ALSO touched (a genuine multi-category fix):
        multilabel=True -> 1.0 (full credit; the call correctly describes part
                           of the fix), partial=True -> 0.5, else 0.0.
    - 0.0 if the predicted bucket is one the fix did NOT touch at all.
    - None if NOT_SCORABLE.

    multilabel is the most generous DEFENSIBLE setting (it never credits a
    category the fix didn't touch); it lifts BOTH systems, so report which you used.
    """
    primary, allb = substantive_fix_buckets(reasoning)
    if primary is None:
        return None
    pb = pred_bucket(category)
    if pb == primary:
        return 1.0
    if pb in allb:
        return 1.0 if multilabel else (0.5 if partial else 0.0)
    return 0.0


# ── Loading / dedup ────────────────────────────────────────────────────────

def load_results(paths) -> list[dict]:
    byid = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            byid[r.get("run_id", id(r))] = r   # last write wins (newest run)
    return list(byid.values())


def _cat(rec, side):
    pred = rec.get(side, {}).get("prediction") or {}
    return pred.get("category") if isinstance(pred, dict) else None


# ── Scoring ────────────────────────────────────────────────────────────────

def score(rows: list[dict], balanced: bool = False, seed: int = 0):
    # Keep only cases with a scorable GT bucket.
    scorable = [r for r in rows if gt_bucket(r.get("ground_truth", {}).get("action")) is not None]

    if balanced:
        by_b = collections.defaultdict(list)
        for r in scorable:
            by_b[gt_bucket(r["ground_truth"]["action"])].append(r)
        n = min(len(v) for v in by_b.values()) if by_b else 0
        rng = random.Random(seed)
        picked = []
        for b in BUCKETS:
            if by_b.get(b):
                picked.extend(rng.sample(by_b[b], min(n, len(by_b[b]))))
        scorable = picked
        print(f"[balanced] equal-n subsample: {n} per bucket "
              f"({', '.join(b for b in BUCKETS if by_b.get(b))})")

    per = {b: {"n": 0, "rpa": 0, "apa": 0} for b in BUCKETS}
    div = {"n": 0, "apa_right": 0, "rpa_right": 0, "both_wrong": 0}
    for r in scorable:
        gtb = gt_bucket(r["ground_truth"]["action"])
        rv = coarse_verdict(_cat(r, "rpa"), r["ground_truth"]["action"])
        av = coarse_verdict(_cat(r, "apa"), r["ground_truth"]["action"])
        d = per[gtb]
        d["n"] += 1
        d["rpa"] += rv == "CORRECT"
        d["apa"] += av == "CORRECT"
        if pred_bucket(_cat(r, "rpa")) != pred_bucket(_cat(r, "apa")):
            div["n"] += 1
            if av == "CORRECT" and rv != "CORRECT":
                div["apa_right"] += 1
            elif rv == "CORRECT" and av != "CORRECT":
                div["rpa_right"] += 1
            else:
                div["both_wrong"] += 1

    used = {b: d for b, d in per.items() if d["n"]}
    tot_n = sum(d["n"] for d in used.values())
    tot_r = sum(d["rpa"] for d in used.values())
    tot_a = sum(d["apa"] for d in used.values())

    print(f"\nscorable cases: {tot_n}")
    print(f"\n{'coarse GT bucket':14} {'n':>4} {'RPA':>11} {'APA':>11}")
    for b in BUCKETS:
        if b in used:
            d = used[b]
            print(f"{b:14} {d['n']:>4} {d['rpa']:>3} ={d['rpa']/d['n']:>6.0%} "
                  f"{d['apa']:>3} ={d['apa']/d['n']:>6.0%}")
    if tot_n:
        print(f"{'MICRO (overall)':14} {tot_n:>4} {tot_r:>3} ={tot_r/tot_n:>6.0%} "
              f"{tot_a:>3} ={tot_a/tot_n:>6.0%}")
        macro_r = sum(d["rpa"] / d["n"] for d in used.values()) / len(used)
        macro_a = sum(d["apa"] / d["n"] for d in used.values()) / len(used)
        print(f"{'MACRO (per-cls)':14} {'':>4} {'':>3}  {macro_r:>5.0%} {'':>3}  {macro_a:>5.0%}"
              f"   <- imbalance-robust headline")
        print(f"\n-- when RPA & APA predict DIFFERENT buckets (n={div['n']}) --")
        print(f"  APA right, RPA wrong: {div['apa_right']}   <- APA's added value")
        print(f"  RPA right, APA wrong: {div['rpa_right']}   <- APA's regressions")
        print(f"  both wrong/other:     {div['both_wrong']}")


def _load_expert_revisions():
    """Documented manual ground-truth corrections (expert adjudication). Maps
    run_id -> {"action": "NOT_SCORABLE"|"<BUCKET>", "reason": ...}. Applied
    uniformly (selected by file evidence, not by which system it helps)."""
    import os
    p = "data/expert_revisions.json"
    if not os.path.exists(p):
        return {}
    return json.loads(Path(p).read_text(encoding="utf-8"))


def score_substantive(rows, partial: bool = True, exclude_dependency: bool = False,
                      multilabel: bool = False):
    """Score against the SUBSTANTIVE fix bucket (lockfile noise ignored), with
    partial credit. exclude_dependency drops the DEPENDENCY class, whose
    file-based labels are documented as unreliable (see substantive_fix_buckets).
    Both modes MUST be disclosed in any writeup that quotes these numbers.
    """
    revisions = _load_expert_revisions()
    n_revised = 0
    per = {b: {"n": 0, "rpa": 0.0, "apa": 0.0} for b in BUCKETS}
    for r in rows:
        rev = revisions.get(r.get("run_id", ""))
        if rev and rev.get("action") == "NOT_SCORABLE":
            n_revised += 1
            continue                      # expert-excluded: mislabeled ground truth
        primary, _ = substantive_fix_buckets(r.get("ground_truth", {}).get("reasoning"))
        if rev and rev.get("action") in BUCKETS:
            primary = rev["action"]       # expert-corrected bucket
        if primary is None:
            continue
        if exclude_dependency and primary == "DEPENDENCY":
            continue
        reason = r["ground_truth"].get("reasoning")
        per[primary]["n"] += 1
        per[primary]["rpa"] += substantive_credit(_cat(r, "rpa"), reason, partial, multilabel) or 0.0
        per[primary]["apa"] += substantive_credit(_cat(r, "apa"), reason, partial, multilabel) or 0.0

    used = {b: d for b, d in per.items() if d["n"]}
    if not used:
        print("no scorable cases"); return
    tag = "multilabel" if multilabel else ("partial" if partial else "strict")
    excl = "  (DEPENDENCY excluded: unreliable labels)" if exclude_dependency else ""
    rev_note = f"  [{n_revised} expert-revised → NOT_SCORABLE]" if n_revised else ""
    print(f"\nSUBSTANTIVE re-bucketing, {tag} credit{excl}{rev_note}")
    print(f"{'bucket':14} {'n':>4} {'RPA':>11} {'APA':>11}")
    for b in BUCKETS:
        if b in used:
            d = used[b]
            print(f"{b:14} {d['n']:>4} {d['rpa']:>5.1f}={d['rpa']/d['n']:>5.0%} "
                  f"{d['apa']:>5.1f}={d['apa']/d['n']:>5.0%}")
    macro_r = sum(d["rpa"] / d["n"] for d in used.values()) / len(used)
    macro_a = sum(d["apa"] / d["n"] for d in used.values()) / len(used)
    print(f"{'MACRO':14} {'':>4} {'':>5} {macro_r:>5.0%} {'':>5} {macro_a:>5.0%}"
          f"   <- RPA vs APA (imbalance-robust)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="one or more eval_500_results*.jsonl files")
    ap.add_argument("--balanced", action="store_true",
                    help="subsample to equal n per coarse bucket before scoring")
    ap.add_argument("--substantive", action="store_true",
                    help="re-bucket GT by substantive fix files (lockfile-noise-aware), partial credit")
    ap.add_argument("--exclude-dep", action="store_true",
                    help="with --substantive: drop the DEPENDENCY class (unreliable labels; MUST disclose)")
    ap.add_argument("--strict", action="store_true", help="with --substantive: no partial credit")
    ap.add_argument("--multilabel", action="store_true",
                    help="with --substantive: FULL credit for any category the fix actually touched "
                         "(most generous defensible scoring; lifts both systems)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = load_results(args.results)
    print(f"loaded {len(rows)} unique cases from {len(args.results)} file(s)")
    if args.substantive:
        score_substantive(rows, partial=not args.strict, exclude_dependency=False, multilabel=args.multilabel)
        score_substantive(rows, partial=not args.strict, exclude_dependency=True, multilabel=args.multilabel)
    else:
        score(rows, balanced=args.balanced, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
