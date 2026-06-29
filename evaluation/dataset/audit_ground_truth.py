#!/usr/bin/env python3
"""
audit_ground_truth.py
=====================
Human spot-check sheet for the ground-truth labels.

Runs the full ground-truth scraper (with the relevance filter) on a random
sample of the evaluation dataset and writes a sheet you can verify by hand:
for each case it shows the failure, the follow-up commit/PR the scraper picked
as "the fix", the auto-assigned label, the relevance reason, and a GitHub link
to the failing commit so you can click through and confirm.

Fill in the `human_label` / `human_correct` columns yourself, then the script's
companion summary (run with --summarize) reports the agreement rate to cite in
the thesis as a ground-truth-quality measure.

Usage:
  export GITHUB_TOKEN=...            # higher rate limit
  python audit_ground_truth.py --n 50
  # ...verify rows in data/gt_audit.csv by hand (set human_correct to 1/0)...
  python audit_ground_truth.py --summarize
"""

from __future__ import annotations

# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------

import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

sys.path.insert(0, ".")

DATA = Path("data/dataset_diverse_500.jsonl.gz")
OUT = Path("data/gt_audit.csv")
FIELDS = [
    "idx", "repo", "branch", "failing_sha", "commit_title",
    "auto_label", "method", "relevance_reasons", "n_followups",
    "fix_commit_or_pr", "fix_message", "github_link",
    "auto_reasoning",
    "human_label", "human_correct",   # <- you fill these
]


def load_cases():
    cases = []
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_audit(n: int, seed: int):
    from src.apa.intake_parser import RunEvent
    from archive.ground_truth_scraper import scrape_ground_truth

    cases = load_cases()
    random.seed(seed)
    sample = random.sample(cases, min(n, len(cases)))
    print(f"Auditing {len(sample)} of {len(cases)} cases (seed={seed})")

    rows = []
    for i, c in enumerate(sample, 1):
        ik = c.get("intake", {})
        # Build a minimal RunEvent the scraper needs.
        ev = RunEvent(
            source="github", run_id=ik.get("run_id", ""), repo=ik.get("repo", ""),
            event=ik.get("event", ""), conclusion=ik.get("conclusion", "failure"),
            started_at=ik.get("started_at", "") or c.get("metadata", {}).get("scan_line", ""),
            n_jobs=ik.get("n_jobs", 0), failed_jobs_count=ik.get("failed_jobs_count", 0),
            duration_sec=None, branch=ik.get("branch", ""),
            commit_sha=ik.get("commit_sha", ""), commit_title=ik.get("commit_title", ""),
            commit_author=ik.get("commit_author", ""), workflow=ik.get("workflow", ""),
        )
        repo = ev.repo
        print(f"  [{i}/{len(sample)}] {repo} @ {ev.commit_sha[:10]}...")
        try:
            gt = scrape_ground_truth(ev)
        except Exception as e:
            print(f"      scrape failed: {e}")
            continue

        fups = gt.follow_up_commits or []
        first = fups[0] if fups else None
        link = f"https://github.com/{repo}/commit/{ev.commit_sha}" if ev.commit_sha else ""
        rows.append({
            "idx": i,
            "repo": repo,
            "branch": ev.branch,
            "failing_sha": ev.commit_sha[:12],
            "commit_title": (ev.commit_title or "")[:80],
            "auto_label": gt.developer_action,
            "method": gt.classification_method,
            "relevance_reasons": ",".join(gt.signals.get("relevance_reasons", []))
                                  if isinstance(gt.signals, dict) else "",
            "n_followups": gt.n_follow_up_commits,
            "fix_commit_or_pr": (first.sha if first else
                                 gt.signals.get("pr_number", "") if isinstance(gt.signals, dict) else ""),
            "fix_message": (first.message_title if first else "")[:80],
            "github_link": link,
            "auto_reasoning": (gt.developer_action_reasoning or "")[:200],
            "human_label": "",
            "human_correct": "",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # Quick label distribution so you see balance before verifying.
    from collections import Counter
    dist = Counter(r["auto_label"] for r in rows)
    print(f"\nWrote {len(rows)} rows to {OUT}")
    print("Auto-label distribution:", dict(dist))
    scorable = sum(1 for r in rows if r["auto_label"] in
                   ("CODE_FIX", "WORKFLOW_FIX", "PIN_VERSION", "REVERT",
                    "DEPENDENCY_CHANGE"))
    print(f"Scorable (has a fix label): {scorable}/{len(rows)}")
    print("\nNow open the CSV and fill 'human_correct' (1=correct, 0=wrong) "
          "for each row by clicking the github_link, then run --summarize.")


def summarize():
    if not OUT.exists():
        print(f"No audit file at {OUT}. Run without --summarize first.")
        return
    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    checked = [r for r in rows if str(r.get("human_correct", "")).strip() in ("0", "1")]
    if not checked:
        print("No rows verified yet (human_correct is empty). Fill it in first.")
        return
    correct = sum(1 for r in checked if r["human_correct"].strip() == "1")
    n = len(checked)
    print(f"Ground-truth audit: {correct}/{n} labels verified correct "
          f"({100*correct/n:.0f}%)")
    # Break down by auto_label so you can see which labels are least reliable.
    from collections import defaultdict
    by = defaultdict(lambda: [0, 0])
    for r in checked:
        by[r["auto_label"]][0] += int(r["human_correct"].strip() == "1")
        by[r["auto_label"]][1] += 1
    print("By label:")
    for lbl, (c, t) in sorted(by.items()):
        print(f"  {lbl:20} {c}/{t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        run_audit(args.n, args.seed)


if __name__ == "__main__":
    main()
