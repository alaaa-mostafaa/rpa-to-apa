# test_three_hard_cases.py
# Run intake + log_extractor on three deliberately-different hard
# cases so we see whether the extractor generalizes across:
#   - language ecosystems (Java, Python wheels, R)
#   - log size (5.7 MB, 500 KB, 465 KB)
#   - job structure (multi-job, multi-platform matrix, single-job)

import gzip
import json
from pathlib import Path

from src.apa.intake_parser import intake
from src.apa.log_extractor import extract_log_excerpt

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"

CASES = [
    {
        "label": "X — apache/unomi (Java, 5.7 MB, 4 jobs)",
        "run_id": "apache/unomi_.github/workflows/unomi-ci-build-tests.yml_1369_1",
    },
    {
        "label": "Y — pyca/bcrypt (Python wheel matrix, 8 jobs / 81 steps)",
        "run_id": "pyca/bcrypt_.github/workflows/wheel-builder.yml_82_1",
    },
    {
        "label": "Z — ohdsi/dataqualitydashboard (R CMD check, 465 KB)",
        "run_id": "ohdsi/dataqualitydashboard_.github/workflows/R_CMD_check_main_weekly.yaml_28_1",
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
    return found


def tarball_path_in_zip(raw):
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/"):] if p.startswith("/data/") else p


def short(s, n):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def run_case(label, raw):
    print()
    print("█" * 90)
    print(f"  CASE {label}")
    print("█" * 90)

    meta = raw.get("metadata") or {}
    head_commit = meta.get("head_commit") or {}
    insights = raw.get("log_insights") or []
    print(f"\n  Repo:    {raw.get('repository_name')}")
    print(f"  Branch:  {meta.get('head_branch')}")
    print(f"  Commit:  {short(head_commit.get('message', ''), 90)}")
    print(f"  Jobs:    {len(insights)}  "
          f"Total steps: {sum(len(j.get('steps') or []) for j in insights)}  "
          f"Log size: {sum(j.get('log_size', 0) for j in insights):,} bytes")

    event = intake(raw)
    print()
    print(f"  ── INTAKE ──")
    print(f"  detection mode: {event.failure_detection}")
    print(f"  failed jobs:    {event.failed_jobs_count}")
    print(f"  failed steps:")
    for fs in event.failed_steps:
        print(f"     - job_file:  {fs.job_file}")
        print(f"       step:      #{fs.step_index}  ({fs.step_type})")
        print(f"       label:     {short(fs.step_label, 70)}")
        print(f"       detection: {fs.detection_mode}")

    if not event.failed_steps:
        print("\n  (no failed steps — nothing to extract)")
        return

    tarball = tarball_path_in_zip(raw)

    print()
    print(f"  ── EXTRACT (per failed step) ──")
    for i, fs in enumerate(event.failed_steps, 1):
        excerpt = extract_log_excerpt(
            zip_path=ZIP_PATH,
            tarball_name=tarball,
            job_file=fs.job_file,
            step_label=fs.step_label or "",
        )
        print(f"\n  [step {i}/{len(event.failed_steps)}] {fs.job_file}")
        print(f"     strategy:        {excerpt.strategy_used}")
        print(f"     total file lines: {excerpt.total_lines_in_file:,}")
        print(f"     ##[error] count: {len(excerpt.error_marker_lines)}")
        print(f"     windows:         {len(excerpt.error_windows)}")
        print(f"     lines kept:      {sum(len(w) for w in excerpt.error_windows):,}")
        print(f"     truncated:       {excerpt.truncated}")
        if excerpt.extraction_note:
            print(f"     note:            {excerpt.extraction_note}")

        if excerpt.error_marker_lines:
            print(f"     first marker:    {short(excerpt.error_marker_lines[0], 80)}")

        # Show LAST 25 lines of the LAST error window — usually
        # where the readable error message lives.
        if excerpt.error_windows:
            last_window = excerpt.error_windows[-1]
            print()
            print(f"     ── last 25 lines of last window ──")
            for ln in last_window[-25:]:
                print(f"       {ln}")

        # Save full excerpt
        safe_id = raw["_id"].replace("/", "_").replace(".", "_")
        out_dir = Path("/home/guc_alaa/hard_case_excerpts")
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"{safe_id}__step{i}.txt"
        out.write_text(excerpt.as_prompt_text(), encoding="utf-8")
        print(f"\n     ✓ Full excerpt: {out}")


def main():
    runs = find_runs([c["run_id"] for c in CASES])
    print(f"Found {len(runs)}/{len(CASES)} runs.\n")

    for case in CASES:
        raw = runs.get(case["run_id"])
        if not raw:
            print(f"\n!! MISSING: {case['run_id']}")
            continue
        run_case(case["label"], raw)


if __name__ == "__main__":
    main()