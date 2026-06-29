# test_batch_extraction.py
# Run intake + log_extractor on multiple candidate runs to see how
# the pipeline behaves across different repos and failure types.
# Prints one-line summaries so we can spot patterns and outliers.

import gzip
import json
from pathlib import Path

from src.apa.intake_parser import intake
from src.apa.log_extractor import extract_log_excerpt

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
ZIP_PATH = "/home/guc_alaa/github_run_logs.zip"

# A fixed list of known-interesting runs (historically collected by older extractor scripts).
TARGET_RUN_IDS = [
    "devfile/api_.github/workflows/codecov.yaml_23_1",
    "devfile/api_.github/workflows/codecov.yaml_24_1",
    "devfile/api_.github/workflows/release-typescript-models.yaml_109_1",
    "devfile/api_.github/workflows/release-typescript-models.yaml_110_1",
    "wix/react-native-notifications_.github/workflows/documentation.yml_274_1",
    "wix/react-native-notifications_.github/workflows/documentation.yml_275_1",
    "wix/react-native-notifications_.github/workflows/documentation.yml_276_1",
    "wix/react-native-notifications_.github/workflows/documentation.yml_277_1",
    "wix/react-native-notifications_.github/workflows/documentation.yml_278_1",
    "nerdwalletoss/shepherd_.github/workflows/ci.yml_667_1",
    "nerdwalletoss/shepherd_.github/workflows/ci.yml_668_1",
]


def tarball_path_in_zip(raw: dict) -> str:
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/"):] if p.startswith("/data/") else p


def short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def find_runs(target_ids: list) -> dict:
    """Stream runs.json.gz once, collecting all target runs by _id."""
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
    runs = find_runs(TARGET_RUN_IDS)

    print("=" * 110)
    print(f"{'#':<3} {'repo':<35} {'wf':<22} {'strategy':<16} {'lines':<7} {'markers':<8} {'first error':<30}")
    print("=" * 110)

    for i, rid in enumerate(TARGET_RUN_IDS, 1):
        raw = runs.get(rid)
        if not raw:
            print(f"{i:<3} (not found in metadata)")
            continue

        event = intake(raw)
        tarball = tarball_path_in_zip(raw)
        repo = short(event.repo, 33)
        workflow = short(event.workflow, 20)

        if not event.failed_steps:
            print(f"{i:<3} {repo:<35} {workflow:<22} {'NO_FAILED_STEPS':<16}")
            continue

        # Use the first failed step (most cases here are single-job)
        fs = event.failed_steps[0]
        excerpt = extract_log_excerpt(
            zip_path=ZIP_PATH,
            tarball_name=tarball,
            job_file=fs.job_file,
            step_label=fs.step_label or "",
        )

        total_lines = sum(len(w) for w in excerpt.error_windows)
        n_markers = len(excerpt.error_marker_lines)
        first_marker = ""
        if excerpt.error_marker_lines:
            first_marker = short(excerpt.error_marker_lines[0].strip(), 28)
        elif excerpt.extraction_note:
            first_marker = f"({short(excerpt.extraction_note, 26)})"

        print(
            f"{i:<3} {repo:<35} {workflow:<22} "
            f"{excerpt.strategy_used:<16} {total_lines:<7} {n_markers:<8} {first_marker:<30}"
        )

        # Save each excerpt
        safe_id = rid.replace("/", "_").replace(".", "_")
        out_path = Path(f"/home/guc_alaa/batch_excerpts/{safe_id}.txt")
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(excerpt.as_prompt_text(), encoding="utf-8")

    print("=" * 110)
    print(f"\nFull excerpts saved to /home/guc_alaa/batch_excerpts/")


if __name__ == "__main__":
    main()