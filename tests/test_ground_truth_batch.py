# test_ground_truth_batch.py
# Run the ground truth scraper on multiple cases and print a summary.

import gzip
import json
from pathlib import Path
from dataclasses import asdict

from src.apa.intake_parser import intake
from archive.ground_truth_scraper import scrape_ground_truth, print_ground_truth

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")

CASES = [
    {
        "label": "pyca/bcrypt — dependabot checkout bump",
        "run_id": "pyca/bcrypt_.github/workflows/wheel-builder.yml_82_1",
    },
    {
        "label": "hibernate/hibernate-search — ORM upgrade",
        "run_id": "hibernate/hibernate-search_.github/workflows/simple-build.yml_66_1",
    },
    {
        "label": "apache/unomi — documentation fix",
        "run_id": "apache/unomi_.github/workflows/unomi-ci-build-tests.yml_1369_1",
    },
    {
        "label": "nerdwalletoss/shepherd — renovate lodash",
        "run_id": "nerdwalletoss/shepherd_.github/workflows/ci.yml_667_1",
    },
    {
        "label": "devfile/api — codecov",
        "run_id": "devfile/api_.github/workflows/codecov.yaml_23_1",
    },
]


def find_runs(target_ids):
    wanted = set(target_ids)
    found = {}
    print(f"Streaming runs.json.gz to find {len(wanted)} runs...")
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = run.get("_id")
            if rid in wanted:
                found[rid] = run
                if len(found) == len(wanted):
                    break
    print(f"Found {len(found)}/{len(wanted)} runs.\n")
    return found


def main():
    runs = find_runs([c["run_id"] for c in CASES])

    out_dir = Path("/home/guc_alaa/ground_truth_results")
    out_dir.mkdir(exist_ok=True)

    summary = []

    for case in CASES:
        print()
        print("=" * 80)
        print(f"  {case['label']}")
        print("=" * 80)

        raw = runs.get(case["run_id"])
        if not raw:
            print("  NOT FOUND in metadata")
            continue

        event = intake(raw)
        print(f"  repo:   {event.repo}")
        print(f"  branch: {event.branch}")
        print(f"  commit: {event.commit_title}")

        gt = scrape_ground_truth(event)
        print_ground_truth(gt)

        summary.append({
            "label": case["label"],
            "repo": gt.repo,
            "branch": gt.branch,
            "repo_accessible": gt.repo_accessible,
            "branch_found": gt.branch_found,
            "follow_ups": gt.n_follow_up_commits,
            "developer_action": gt.developer_action,
            "reasoning": gt.developer_action_reasoning,
        })

        # Save full result
        safe_id = case["run_id"].replace("/", "_").replace(".", "_")
        out_path = out_dir / f"{safe_id}.json"
        out_path.write_text(
            json.dumps(asdict(gt), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  saved: {out_path}")

    # Print summary table
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for s in summary:
        status = "accessible" if s["repo_accessible"] else "NOT ACCESSIBLE"
        branch = "branch exists" if s["branch_found"] else "branch gone"
        print(f"\n  {s['label']}")
        print(f"    repo: {status} | {branch} | follow-ups: {s['follow_ups']}")
        print(f"    DEVELOPER ACTION: {s['developer_action']}")
        print(f"    {s['reasoning'][:120]}")


if __name__ == "__main__":
    main()