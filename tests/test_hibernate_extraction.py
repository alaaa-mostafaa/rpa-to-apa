# test_hibernate_extraction.py
# End-to-end test on a real failing run: hibernate/hibernate-search
# run 66, "Upgrade to ORM 6.2.9".
#
# Pipeline:
#   1. Stream runs.json.gz, find this specific run.
#   2. Run intake_parser on it → RunEvent.
#   3. For each failed step in the RunEvent, run log_extractor on the
#      corresponding tarball entry from the 142 GB zip.
#   4. Print a clean summary so we can see whether the log text
#      actually contains useful error information.
#
# No LLM call. Just deterministic plumbing on real data.

import gzip
import json
from pathlib import Path

from src.apa.intake_parser import intake, pretty_print
from src.apa.log_extractor import extract_log_excerpt

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"
TARGET_RUN_ID = "hibernate/hibernate-search_.github/workflows/simple-build.yml_66_1"


def find_run(run_id: str) -> dict:
    """Stream runs.json.gz to locate a specific run by _id."""
    print(f"Searching for {run_id} in runs.json.gz ...")
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run.get("_id") == run_id:
                return run
    raise SystemExit(f"Run {run_id} not found.")


def tarball_path_in_zip(raw: dict) -> str:
    """Convert /data/logs/... to logs/..."""
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/"):] if p.startswith("/data/") else p


def main():
    # ── 1. Find the run in the metadata ─────────────────────────────
    raw_run = find_run(TARGET_RUN_ID)
    print(f"✓ Found run.\n")

    # ── 2. Intake → RunEvent ────────────────────────────────────────
    print("─" * 70)
    print("STAGE 1: INTAKE PARSER")
    print("─" * 70)
    event = intake(raw_run)
    pretty_print(event, raw_run=raw_run)

    if not event.failed_steps:
        print("\n(no failed steps detected — nothing to extract)")
        return

    # ── 3. For each failed step, run the log extractor ──────────────
    tarball = tarball_path_in_zip(raw_run)
    print()
    print("─" * 70)
    print("STAGE 2: LOG EXTRACTOR")
    print("─" * 70)
    print(f"  tarball: {tarball}\n")

    for i, fs in enumerate(event.failed_steps, 1):
        print(f"\n══ Failed step [{i}/{len(event.failed_steps)}] ══")
        print(f"   job_file:    {fs.job_file}")
        print(f"   step_label:  {fs.step_label}")
        print(f"   detection:   {fs.detection_mode}")
        print()

        excerpt = extract_log_excerpt(
            zip_path=ZIP_PATH,
            tarball_name=tarball,
            job_file=fs.job_file,
            step_label=fs.step_label or "",
        )

        print(f"   ── extraction result ──")
        print(f"   total lines in file:     {excerpt.total_lines_in_file:,}")
        print(f"   group found:             {excerpt.group_found}")
        print(f"   group lines kept:        {len(excerpt.group_lines):,}")
        print(f"   group truncated:         {excerpt.group_truncated}")
        print(f"   ##[error] markers found: {len(excerpt.error_marker_lines)}")
        if excerpt.extraction_note:
            print(f"   note:                    {excerpt.extraction_note}")

        if excerpt.error_marker_lines:
            print()
            print("   ── ##[error] markers ──")
            for ln in excerpt.error_marker_lines[:10]:
                print(f"     {ln}")

        if excerpt.group_lines:
            print()
            print("   ── last 30 lines of group block (where errors usually live) ──")
            for ln in excerpt.group_lines[-30:]:
                print(f"     {ln}")

        # Save full excerpt for inspection
        out = Path(f"/home/guc_alaa/hibernate_excerpt_step{i}.txt")
        out.write_text(excerpt.as_prompt_text(), encoding="utf-8")
        print(f"\n   ✓ Full excerpt saved to {out}")


if __name__ == "__main__":
    main()