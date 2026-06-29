#!/usr/bin/env python3
"""
build_dataset.py
================
Diversity-first builder for the curated CI-failure evaluation corpus.

Replaces the ad-hoc extractor: this version makes repository diversity the
primary design goal. Every retained case must satisfy four hard inclusion
gates, and per-repository / per-workflow caps are enforced so that no single
repository can dominate the corpus.

Inclusion gates (all deterministic, no LLM):
  1. Terminal FAILURE conclusion (success/cancelled/skipped excluded).
  2. An accessible log archive path (so logs can be extracted downstream).
  3. Execution-context filter: protected/important branch, at least one job
     with >= MIN_STEPS_PER_JOB steps and at least one shell step, and the
     failure is a real failure (not purely a GHALogs tooling artifact).
  4. A verifiable ground-truth remediation action, recovered with ONE GitHub
     API call (quick_gt_preflight). Cases with no timely developer response
     are dropped here so the corpus is fully scorable by construction.

Diversity controls:
  --max-per-repo           hard cap on cases from any single repository
                           (default 1 -> at most one case per repo).
  --max-per-repo-workflow  cap per (repo, workflow) pair (default 1).

The scan is a single sequential pass with a fixed range, so re-running with
the same arguments regenerates the same corpus (modulo live GitHub state for
the ground-truth gate). No randomization is used.

Typical use (1 case/repo, ground-truth-verified, log excerpts embedded):
  python build_dataset.py \
      --runs-path /home/guc_alaa/runs.json.gz \
      --zip-path  /home/guc_alaa/github_run_logs.zip \
      --out data/dataset_diverse.jsonl.gz --format jsonl.gz \
      --n 5000 --max-per-repo 1 --max-per-repo-workflow 1 \
      --include-log-excerpts
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
import gzip
import json
import os
import re
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── constants ─────────────────────────────────────────────────────────

TOOLING_PATTERNS = (
    "bash-command-extractor",
    "Converting circular structure to JSON",
    "BashWord",
    "Parser exception",
)

PROTECTED_BRANCHES = {"main", "master", "develop", "dev", "release"}

# Ground-truth actions that count as a verifiable developer fix. Anything
# else (UNKNOWN, NO_IMMEDIATE_RESPONSE, AMBIGUOUS, PR_CLOSED_UNMERGED, ...)
# is treated as not-scorable and dropped at the ground-truth gate.
SCORABLE_ACTIONS = {
    "CODE_FIX",
    "WORKFLOW_FIX",
    "PIN_VERSION",
    "REVERT",
    "DEPENDENCY_CHANGE",
    "PR_MERGED",
}

_SECRET_PATTERNS = [
    (r"\bghp_[A-Za-z0-9]{30,}\b", "ghp_[REDACTED]"),
    (r"\bgithub_pat_[A-Za-z0-9_]{30,}\b", "github_pat_[REDACTED]"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AKIA[REDACTED]"),
    (r"(?i)\bBearer\s+[A-Za-z0-9._\-]+=*\b", "Bearer [REDACTED]"),
    (r"(?i)(API[_-]?KEY\s*[=:]\s*)([^\s'\"]+)", r"\1[REDACTED]"),
    (r"(?i)(SECRET\s*[=:]\s*)([^\s'\"]+)", r"\1[REDACTED]"),
    (r"(?i)(TOKEN\s*[=:]\s*)([^\s'\"]+)", r"\1[REDACTED]"),
    (r"(?i)(PASSWORD\s*[=:]\s*)([^\s'\"]+)", r"\1[REDACTED]"),
]


# ── small helpers ─────────────────────────────────────────────────────

def _redact_secrets(text: str) -> str:
    if not text:
        return text
    out = text
    for pat, repl in _SECRET_PATTERNS:
        out = re.sub(pat, repl, out)
    return out


def is_tooling_artifact(payload: object) -> bool:
    if not payload:
        return False
    text = json.dumps(payload) if not isinstance(payload, str) else payload
    return any(p in text for p in TOOLING_PATTERNS)


# Markers that, by themselves, carry NO diagnostic content. A log made up of
# only these lines tells you the run failed but not WHY.
_UNINFORMATIVE_LOG_MARKERS = (
    "process completed with exit code",
    "the operation was canceled",
    "the operation was cancelled",
    "##[error]process completed",
    "exited with code",
)

# Lines that are NOISE — they show ACTIVITY (checkout, build progress, env echoes),
# not a diagnosis. A log made of only these tells you nothing about WHY it failed.
# (The previous token list was a no-op: it accepted any line containing "error",
# "test", "version", ".js" etc., which every CI log has — so it passed 100% of
# cases. This rewrite requires an actual error/failure MESSAGE.)
_NOISE_LINE_RE = re.compile(
    r"^(===|---|##\[(group|endgroup|section|command|debug)\]|failing step|"
    r"extraction strategy|shell:|env:|with:|\[command\]|remote:|\*\s*\[new ref\]|"
    r"reset branch|(switched to|already on) |set up to track|head is now at|"
    r"(receiving|resolving|counting|compressing|enumerating|unpacking) (objects|deltas)|"
    r"entering '|cloning into|submodule|from https?://|"
    r"(compiling|downloading|collecting|installing|building wheel|preparing metadata|"
    r"requirement already satisfied|successfully installed|fetching|using cached|"
    r"npm warn|warning: ) )",
    re.I,
)

# Strong, specific signals of a REAL, diagnosable failure (not generic activity).
_REAL_ERROR_RE = re.compile(
    r"error:\s|error\]|\d+\s+errors?\b|errors?\s+(?:found|occurred|encountered)|"
    r"error compiling|compilation error|build failed|"
    r"\btests?\s+failed\b|test failure|^FAILED\b|\bFAILED\s+\w|\d+\s+failed\b|"
    r"traceback \(most recent call last\)|assertionerror|"
    r"\bexception\b|\bpanic:|\bfatal:\s|fatal error|segmentation fault|core dumped|"
    r"syntaxerror|typeerror|referenceerror|nameerror|importerror|modulenotfounderror|"
    r"cannot find|could not find|command not found|no such file|could not resolve|"
    r"no matching version|version conflict|dependency resolution|npm err!|"
    r"\b[EWF]\d{3,4}\s|"
    r"(?<![_a-z])unauthorized\b|permission denied|access denied|"
    r"invalid workflow|bad yaml|yaml.{0,20}error|"
    r"timed out\b|connection refused|rate limit|unresolved (?:import|reference)",
    re.I,
)


def is_informative_log(excerpt: str) -> bool:
    """True only when the failure log contains an ACTUAL error/failure message.

    Strips the extractor scaffolding plus pure-activity noise (git checkout,
    build progress, env echoes, npm deprecation warnings) and the generic
    terminal markers, then requires at least one line carrying a real diagnostic
    (compile/test error, stack trace, missing module, lint violation, auth/YAML
    error, timeout, etc.). Logs that are only "exit code 1" + activity are
    UNINFORMATIVE — unsolvable from the evidence — so the eval corpus excludes
    them to keep the RPA-vs-APA comparison meaningful.
    """
    text = excerpt or ""
    if not text.strip():
        return False
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        low = ln.lower()
        if any(m in low for m in _UNINFORMATIVE_LOG_MARKERS):
            continue
        if _NOISE_LINE_RE.match(ln):
            continue
        if _REAL_ERROR_RE.search(ln):
            return True
    return False


def is_protected_branch(branch: str) -> bool:
    b = (branch or "").lower().strip()
    return (
        b in PROTECTED_BRANCHES
        or b.startswith("release/")
        or b.startswith("main-")
    )


def tarball_path(raw: dict) -> str:
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/"):] if p.startswith("/data/") else p


def _has_shell_step(insights: list[dict]) -> bool:
    return any(
        step.get("type") == "shell"
        for job in insights
        for step in (job.get("steps") or [])
    )


def _run_started_at(meta: dict) -> str:
    return (
        meta.get("created_at")
        or meta.get("run_started_at")
        or meta.get("started_at")
        or meta.get("updated_at")
        or ""
    )


def _parse_owner_repo(repo_full: str):
    parts = (repo_full or "").split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None, None
    return parts[0], parts[1]


# ── context gate (gate 1-3, deterministic, no network) ────────────────

def passes_context_gate(
    run: dict,
    *,
    protected_only: bool,
    min_steps_per_job: int,
    require_shell_step: bool,
) -> bool:
    meta = run.get("metadata") or {}
    if meta.get("conclusion") != "failure":
        return False

    if protected_only and not is_protected_branch(meta.get("head_branch", "")):
        return False

    if not (run.get("logs_archive") or {}).get("path"):
        return False

    insights = run.get("log_insights") or []
    if not insights:
        return False

    if min_steps_per_job > 0 and not any(
        len(job.get("steps") or []) >= min_steps_per_job for job in insights
    ):
        return False

    if require_shell_step and not _has_shell_step(insights):
        return False

    has_real_error = False
    has_any_error = False
    for job in insights:
        for step in job.get("steps") or []:
            for key in ("error", "errors"):
                if step.get(key):
                    has_any_error = True
                    if not is_tooling_artifact(step.get(key)):
                        has_real_error = True
    if has_any_error and not has_real_error:
        return False

    return True


# ── writer ────────────────────────────────────────────────────────────

class _CaseWriter:
    def __init__(self, path: Path, fmt: str):
        if fmt not in {"jsonl", "jsonl.gz"}:
            raise ValueError("Use --format jsonl or jsonl.gz")
        self.path = path
        self.fmt = fmt
        self._count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        if fmt == "jsonl":
            self._fp = open(self.path, "a", encoding="utf-8")
        else:
            self._fp = gzip.open(self.path, "at", encoding="utf-8")

    def write(self, case: dict) -> None:
        self._fp.write(json.dumps(case, ensure_ascii=False) + "\n")
        self._fp.flush()
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None


# ── main extraction ───────────────────────────────────────────────────

def build(
    *,
    runs_path: Path,
    zip_path: Path,
    out_path: Path,
    n_cases: int,
    scan_start: int,
    scan_end: int,
    protected_only: bool,
    min_steps_per_job: int,
    require_shell_step: bool,
    max_per_repo: int,
    max_per_repo_workflow: int,
    require_ground_truth: bool,
    scorable_actions: set[str],
    require_informative_log: bool,
    include_log_excerpts: bool,
    max_log_chars: int,
    redact: bool,
    max_failed_steps: int,
    progress_every: int,
    output_format: str,
) -> None:
    from src.apa.intake_parser import intake
    from src.apa.log_extractor import extract_log_excerpt

    quick_gt_preflight = None
    if require_ground_truth:
        from archive.ground_truth_scraper import quick_gt_preflight as _q
        quick_gt_preflight = _q

    repo_counts: dict[str, int] = {}
    repo_wf_counts: dict[tuple[str, str], int] = {}

    scanned = 0
    ctx_pass = 0
    gt_checked = 0
    gt_pass = 0
    log_uninformative = 0
    selected = 0

    writer = _CaseWriter(out_path, output_format)

    print(f"Scanning {runs_path}  range [{scan_start:,}, {scan_end:,}]")
    print(f"Target {n_cases} cases | max/repo={max_per_repo} | "
          f"max/(repo,wf)={max_per_repo_workflow} | "
          f"ground_truth_gate={require_ground_truth}")
    print(f"Output: {out_path} ({output_format})")

    with gzip.open(runs_path, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned < scan_start:
                continue
            if scanned > scan_end:
                break
            if progress_every and scanned % progress_every == 0:
                print(f"  scanned {scanned:,} | ctx_pass {ctx_pass} | "
                      f"gt {gt_pass}/{gt_checked} | selected {selected}")

            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Gate 1-3: deterministic context filter.
            if not passes_context_gate(
                run,
                protected_only=protected_only,
                min_steps_per_job=min_steps_per_job,
                require_shell_step=require_shell_step,
            ):
                continue
            ctx_pass += 1

            repo_full = run.get("repository_name", "")
            meta = run.get("metadata") or {}
            branch = meta.get("head_branch", "")
            started_at = _run_started_at(meta)
            sha = (meta.get("head_commit") or {}).get("id", "")
            workflow_name = meta.get("name") or meta.get("workflow") or ""

            owner, repo_name = _parse_owner_repo(repo_full)
            if not owner or not repo_name or not sha or not branch or not started_at:
                continue

            # Diversity caps BEFORE the expensive ground-truth call, so we
            # never spend an API call on a repo/workflow that is already full.
            if max_per_repo and repo_counts.get(repo_full, 0) >= max_per_repo:
                continue
            wf_key = (repo_full, workflow_name)
            if max_per_repo_workflow and repo_wf_counts.get(wf_key, 0) >= max_per_repo_workflow:
                continue

            # Gate 4: verifiable ground-truth remediation action.
            action = "NO_PREFLIGHT"
            if require_ground_truth:
                gt_checked += 1
                try:
                    action = quick_gt_preflight(owner, repo_name, sha, branch, started_at)
                except Exception:
                    continue
                if action not in scorable_actions:
                    continue
                gt_pass += 1

            # Reserve the slot now (case passed every gate).
            repo_counts[repo_full] = repo_counts.get(repo_full, 0) + 1
            repo_wf_counts[wf_key] = repo_wf_counts.get(wf_key, 0) + 1

            # Build the case payload.
            try:
                event = intake(run)
                tarball = tarball_path(run)
                excerpts = []
                sample_errors: list[str] = []

                for fs in event.failed_steps[: max_failed_steps or 1]:
                    ex = extract_log_excerpt(
                        zip_path=str(zip_path),
                        tarball_name=tarball,
                        job_file=fs.job_file,
                        step_label=fs.step_label or "",
                    )
                    excerpts.append(ex)
                    for ml in ex.error_marker_lines[:2]:
                        c = ml.strip()[:120]
                        if c and c not in sample_errors:
                            sample_errors.append(c)

                # Gate 3b: informative-log filter. Drop cases whose failure log
                # carries no real diagnostic (only "exit code 1" / "operation was
                # canceled"). These are unsolvable from the log, so every system
                # collapses to the base rate and the RPA-vs-APA comparison is
                # meaningless. Roll back the reserved diversity slot since the
                # case is rejected here, AFTER the slot was reserved above.
                if require_informative_log:
                    combined_log = "\n".join(ex.as_prompt_text() for ex in excerpts)
                    if not is_informative_log(combined_log):
                        repo_counts[repo_full] -= 1
                        repo_wf_counts[wf_key] -= 1
                        log_uninformative += 1
                        print(f"  ✗ [uninformative_log] {repo_full}  branch={branch}  "
                              f"reason=no_real_error_in_log")
                        continue

                excerpt_payload = None
                if include_log_excerpts:
                    excerpt_payload = []
                    for ex in excerpts:
                        text = ex.as_prompt_text()
                        if redact:
                            text = _redact_secrets(text)
                        if max_log_chars and len(text) > max_log_chars:
                            text = text[: max_log_chars - 1] + "…"
                        excerpt_payload.append({
                            "job_file": getattr(ex, "job_file", ""),
                            "step_label": getattr(ex, "step_label", ""),
                            "strategy": getattr(ex, "strategy_used", ""),
                            "text": text,
                        })

                case = {
                    "case_label": f"{event.repo} — {event.commit_title[:60]}",
                    "preflight_action": action,
                    "intake": {
                        "run_id": event.run_id,
                        "repo": event.repo,
                        "workflow": event.workflow,
                        "branch": event.branch,
                        "event": event.event,
                        "is_protected_branch": event.is_protected_branch,
                        "commit_sha": event.commit_sha,
                        "commit_title": event.commit_title,
                        "conclusion": event.conclusion,
                        "failure_detection": event.failure_detection,
                        "failed_jobs_count": event.failed_jobs_count,
                        "n_jobs": event.n_jobs,
                        "all_failures_are_tooling_artifacts": event.all_failures_are_tooling_artifacts,
                    },
                    "extraction": {
                        "total_steps_extracted": len(excerpts),
                        "strategies": [ex.strategy_used for ex in excerpts],
                        "error_markers_found": sum(len(ex.error_marker_lines) for ex in excerpts),
                        "sample_error_lines": sample_errors[:5],
                    },
                }
                if excerpt_payload is not None:
                    case["extraction"]["log_excerpts"] = excerpt_payload

                writer.write(case)
                selected += 1
                print(f"  ✓ [{selected:>4}/{n_cases}] {repo_full}  "
                      f"branch={branch}  action={action}")
            except Exception as e:
                # Roll back the reserved slot so a build failure does not
                # permanently consume this repo's quota.
                repo_counts[repo_full] -= 1
                repo_wf_counts[wf_key] -= 1
                print(f"  ! build failed for {repo_full}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)

            if selected >= n_cases:
                break

    writer.close()

    print(f"\nDone. Selected {selected} cases from {scanned:,} scanned rows.")
    print(f"  context gate passed: {ctx_pass}")
    if require_ground_truth:
        print(f"  ground-truth gate:   {gt_pass}/{gt_checked} passed")
    if require_informative_log:
        print(f"  dropped (uninformative log): {log_uninformative}")
    print(f"  distinct repositories: {len(repo_counts)}")
    if repo_counts:
        print(f"  max cases in one repo: {max(repo_counts.values())}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-path", type=Path, default=Path("/home/guc_alaa/runs.json.gz"))
    p.add_argument("--zip-path", type=Path, default=Path("/home/guc_alaa/github_run_logs.zip"))
    p.add_argument("--out", type=Path, default=Path("data/dataset_diverse.jsonl.gz"))
    p.add_argument("--format", choices=("jsonl", "jsonl.gz"), default="jsonl.gz")

    p.add_argument("--n", type=int, default=5000, help="Target number of cases")
    p.add_argument("--scan-start", type=int, default=1)
    p.add_argument("--scan-end", type=int, default=600_000)
    p.add_argument("--progress-every", type=int, default=10_000)

    p.add_argument("--no-protected-only", action="store_true",
                   help="Allow non-protected/feature branches")
    p.add_argument("--min-steps-per-job", type=int, default=2)
    p.add_argument("--no-require-shell", action="store_true")
    p.add_argument("--no-require-informative-log", action="store_true",
                   help="Keep cases whose log has no real error (only 'exit code "
                        "1' / 'operation was canceled'). By default these are "
                        "dropped so the corpus is solvable from the log.")

    p.add_argument("--max-per-repo", type=int, default=1,
                   help="Hard cap on cases per repository (default 1)")
    p.add_argument("--max-per-repo-workflow", type=int, default=1,
                   help="Cap per (repo, workflow) pair (default 1)")

    p.add_argument("--no-ground-truth", action="store_true",
                   help="Skip the ground-truth verification gate (corpus will "
                        "NOT be fully scorable; not recommended for the eval set)")
    p.add_argument("--scorable-actions", type=str,
                   default=",".join(sorted(SCORABLE_ACTIONS)),
                   help="Comma-separated developer actions accepted as ground truth")

    p.add_argument("--include-log-excerpts", action="store_true")
    p.add_argument("--max-log-chars", type=int, default=20_000)
    p.add_argument("--no-redact", action="store_true")
    p.add_argument("--max-failed-steps", type=int, default=3)

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    actions = {a.strip() for a in args.scorable_actions.split(",") if a.strip()}
    build(
        runs_path=args.runs_path,
        zip_path=args.zip_path,
        out_path=args.out,
        n_cases=int(args.n),
        scan_start=int(args.scan_start),
        scan_end=int(args.scan_end),
        protected_only=not args.no_protected_only,
        min_steps_per_job=int(args.min_steps_per_job),
        require_shell_step=not args.no_require_shell,
        max_per_repo=int(args.max_per_repo),
        max_per_repo_workflow=int(args.max_per_repo_workflow),
        require_ground_truth=not bool(args.no_ground_truth),
        scorable_actions=actions,
        require_informative_log=not bool(args.no_require_informative_log),
        include_log_excerpts=bool(args.include_log_excerpts),
        max_log_chars=int(args.max_log_chars),
        redact=not bool(args.no_redact),
        max_failed_steps=int(args.max_failed_steps),
        progress_every=int(args.progress_every),
        output_format=str(args.format),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
