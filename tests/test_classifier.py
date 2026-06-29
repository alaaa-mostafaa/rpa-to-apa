# test_classifier.py
# Run the merged classifier on bcrypt with logs, using the new taxonomy.

import gzip
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from src.apa.intake_parser import intake
from src.apa.log_extractor import extract_log_excerpt
from src.apa.classification_agent import classify, print_result

load_dotenv()

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"
RUN_ID = "pyca/bcrypt_.github/workflows/wheel-builder.yml_82_1"


def find_run(run_id):
    print(f"Finding {run_id}...")
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run.get("_id") == run_id:
                return run
    raise SystemExit("Run not found.")


def tarball_path(raw):
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/"):] if p.startswith("/data/") else p


def main():
    from llm_config import make_client
    try:
        client = make_client()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    raw = find_run(RUN_ID)
    event = intake(raw)
    tarball = tarball_path(raw)

    print(f"\nRepo:   {event.repo}")
    print(f"Branch: {event.branch}")
    print(f"Commit: {event.commit_title}")
    print(f"Failed: {event.failed_jobs_count} jobs\n")

    # Extract logs for all failed steps
    print("Extracting log excerpts...")
    excerpts = []
    for fs in event.failed_steps:
        ex = extract_log_excerpt(
            zip_path=ZIP_PATH,
            tarball_name=tarball,
            job_file=fs.job_file,
            step_label=fs.step_label or "",
        )
        excerpts.append(ex)
    print(f"Got {len(excerpts)} excerpts.\n")

    # Classify WITH logs
    print("Calling classifier (with logs + expanded taxonomy)...\n")
    result = classify(event, client, log_excerpts=excerpts)

    print("=" * 70)
    print_result(result)
    print("=" * 70)


if __name__ == "__main__":
    main()