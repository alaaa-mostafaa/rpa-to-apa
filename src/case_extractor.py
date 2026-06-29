from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # Environment variables may already be set; keep running.
    pass

# Ensure friendly UTF-8 output when piping logs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TOOLING_PATTERNS = (
    "bash-command-extractor",
    "Converting circular structure to JSON",
    "BashWord",
    "Parser exception",
)

PROTECTED_BRANCHES = {"main", "master", "develop", "dev", "release"}


def _redact_secrets(text: str) -> str:
    """Best-effort redaction for log text before writing to disk / git.

    This is intentionally conservative and may over-redact.
    """

    import re

    if not text:
        return text

    patterns = [
        # GitHub tokens
        (r"\bghp_[A-Za-z0-9]{30,}\b", "ghp_[REDACTED]"),
        (r"\bgithub_pat_[A-Za-z0-9_]{30,}\b", "github_pat_[REDACTED]"),
        # AWS keys
        (r"\bAKIA[0-9A-Z]{16}\b", "AKIA[REDACTED]"),
        # Generic bearer tokens
        (r"(?i)\bBearer\s+[A-Za-z0-9._\-]+=*\b", "Bearer [REDACTED]"),
        # Common env-style secrets
        (r"(?i)(API[_-]?KEY\s*[=:]\s*)([^\s'\"]+)", r"\1[REDACTED]"),
        (r"(?i)(SECRET\s*[=:]\s*)([^\s'\"]+)", r"\1[REDACTED]"),
        (r"(?i)(TOKEN\s*[=:]\s*)([^\s'\"]+)", r"\1[REDACTED]"),
        (r"(?i)(PASSWORD\s*[=:]\s*)([^\s'\"]+)", r"\1[REDACTED]"),
    ]

    out = text
    for pat, repl in patterns:
        out = re.sub(pat, repl, out)
    return out


class _CaseWriter:
    def __init__(
        self,
        path: Path,
        *,
        fmt: str,
        append: bool,
    ):
        self.path = path
        self.fmt = fmt
        self.append = append

        if fmt not in {"json", "jsonl", "jsonl.gz"}:
            raise ValueError(f"Unsupported format: {fmt}")

        self._count = 0
        self._fp = None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append and self.path.exists():
            self.path.unlink()

        if fmt == "json":
            if append:
                raise ValueError("--append is not supported with --format json")
            self.path.write_text("[]", encoding="utf-8")
        else:
            self._open_for_append()

            if append:
                self._count = self._count_existing_jsonl_records()

    def _open_for_append(self) -> None:
        if self.fmt == "jsonl":
            self._fp = open(self.path, "a", encoding="utf-8")
        elif self.fmt == "jsonl.gz":
            import gzip as _gzip

            self._fp = _gzip.open(self.path, "at", encoding="utf-8")

    def _count_existing_jsonl_records(self) -> int:
        if not self.path.exists():
            return 0
        if self.fmt == "jsonl":
            with open(self.path, "r", encoding="utf-8") as f:
                return sum(1 for ln in f if ln.strip())
        if self.fmt == "jsonl.gz":
            import gzip as _gzip

            with _gzip.open(self.path, "rt", encoding="utf-8") as f:
                return sum(1 for ln in f if ln.strip())
        return 0

    @property
    def count(self) -> int:
        return self._count

    def write_case(self, case: dict) -> None:
        if self.fmt == "json":
            raise RuntimeError("write_case not supported for json")
        if not self._fp:
            raise RuntimeError("writer not open")

        self._fp.write(json.dumps(case, ensure_ascii=False) + "\n")
        self._fp.flush()
        self._count += 1

    def write_full_list(self, cases: list[dict]) -> None:
        if self.fmt != "json":
            raise RuntimeError("write_full_list only supported for json")
        self.path.write_text(
            json.dumps(cases, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def close(self) -> None:
        try:
            if self._fp:
                self._fp.close()
        finally:
            self._fp = None


def is_tooling_artifact(payload: object) -> bool:
    if not payload:
        return False
    text = json.dumps(payload) if not isinstance(payload, str) else payload
    return any(p in text for p in TOOLING_PATTERNS)


def is_protected_branch(branch: str) -> bool:
    b = (branch or "").lower().strip()
    return (
        b in PROTECTED_BRANCHES
        or b.startswith("release/")
        or b.startswith("main-")
        or b.startswith("v")
    )


def tarball_path(raw: dict) -> str:
    p = (raw.get("logs_archive") or {}).get("path", "")
    return p[len("/data/") :] if p.startswith("/data/") else p


def _has_shell_step(insights: list[dict]) -> bool:
    for job in insights:
        for step in job.get("steps") or []:
            if step.get("type") == "shell":
                return True
    return False


def looks_like_real_failure(
    run: dict,
    *,
    protected_only: bool,
    min_steps_per_job: int,
    require_shell_step: bool,
) -> bool:
    meta = run.get("metadata") or {}
    if meta.get("conclusion") != "failure":
        return False

    branch = meta.get("head_branch", "")
    if protected_only and not is_protected_branch(branch):
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
                payload = step.get(key)
                if payload:
                    has_any_error = True
                    if not is_tooling_artifact(payload):
                        has_real_error = True

    # If errors exist but they're all tooling artifacts, skip.
    if has_any_error and not has_real_error:
        return False

    return True


def _run_started_at(meta: dict) -> str:
    return (
        meta.get("created_at")
        or meta.get("run_started_at")
        or meta.get("started_at")
        or meta.get("updated_at")
        or ""
    )


def _parse_owner_repo(repo_full: str) -> tuple[str, str] | tuple[None, None]:
    parts = (repo_full or "").split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None, None
    return parts[0], parts[1]


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def extract_cases(
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
    dedupe_repo: bool,
    preflight_actions: set[str],
    progress_every: int,
    model: str,
    provider: str,
    judge: bool,
    run_preflight: bool,
    output_format: str,
    append: bool,
    include_log_excerpts: bool,
    max_log_chars: int,
    redact: bool,
    run_classification: bool,
    run_ground_truth: bool,
    max_failed_steps: int,
    max_per_repo: int,
    max_per_repo_workflow: int,
) -> list[dict]:
    """Extract high-quality, *scorable* cases to targeted_cases.json format."""

    from src.apa.intake_parser import intake
    from src.apa.log_extractor import extract_log_excerpt

    quick_gt_preflight = None
    scrape_ground_truth = None
    if run_preflight or run_ground_truth:
        from archive.ground_truth_scraper import quick_gt_preflight as _quick_gt_preflight
        from archive.ground_truth_scraper import scrape_ground_truth as _scrape_ground_truth

        quick_gt_preflight = _quick_gt_preflight
        scrape_ground_truth = _scrape_ground_truth

    if judge:
        from evals.evaluation_judge import build_case_summary, llm_judge

    classify = None
    if run_classification:
        from src.apa.classification_agent import classify as _classify

        classify = _classify

    # Only create an LLM client if we will actually call the LLM.
    client = None
    if judge or run_classification:
        from src.apa.llm_config import make_client

        client = make_client(provider=provider)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # For large jsonl/jsonl.gz exports we avoid holding all cases in memory.
    selected: list[dict] = []
    selected_count = 0

    seen_repos: set[str] = set()
    repo_counts: dict[str, int] = {}
    repo_workflow_counts: dict[tuple[str, str], int] = {}

    scanned = 0
    preflight_checked = 0
    preflight_passed = 0

    print(f"Scanning {runs_path} ({scan_start:,}–{scan_end:,}) …")
    print(f"Target: {n_cases} {'scorable ' if run_preflight else ''}cases")
    print(f"Output: {out_path}")
    print(f"Protected only: {protected_only} | dedupe repo: {dedupe_repo}")
    if max_per_repo:
        print(f"Max per repo: {max_per_repo}")
    if max_per_repo_workflow:
        print(f"Max per repo+workflow: {max_per_repo_workflow}")
    if run_preflight:
        print(f"Preflight actions: {sorted(preflight_actions)}")
    else:
        print("Preflight: skipped")
    print(
        f"Provider: {provider} | Model: {model} | judge: {judge} | "
        f"classify: {run_classification} | ground_truth: {run_ground_truth} | "
        f"include_logs: {include_log_excerpts}"
    )

    writer = _CaseWriter(out_path, fmt=output_format, append=append)

    if output_format != "json" and writer.count:
        print(f"Resuming output: {out_path} (already has {writer.count} records)")
        selected_count = writer.count

    if output_format == "json":
        writer.write_full_list(selected)

    with gzip.open(runs_path, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned < scan_start:
                continue
            if scanned > scan_end:
                break

            if progress_every and scanned % progress_every == 0:
                print(
                    f"  scanned {scanned:,} | selected {selected_count} | "
                    f"preflight {preflight_passed}/{preflight_checked} passed"
                )

            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not looks_like_real_failure(
                run,
                protected_only=protected_only,
                min_steps_per_job=min_steps_per_job,
                require_shell_step=require_shell_step,
            ):
                continue

            repo_full = run.get("repository_name", "")
            if dedupe_repo and repo_full in seen_repos:
                continue

            meta = run.get("metadata") or {}
            branch = meta.get("head_branch", "")
            started_at = _run_started_at(meta)
            sha = (meta.get("head_commit") or {}).get("id", "")

            owner, repo_name = _parse_owner_repo(repo_full)
            if not owner or not repo_name or not sha or not branch or not started_at:
                continue

            # Diversity controls (applied before any expensive work).
            # - dedupe_repo=True enforces at most one per repo.
            # - max_per_repo allows a small cap when dedupe_repo=False.
            # - max_per_repo_workflow caps repeats of the same workflow within a repo.
            if not dedupe_repo and max_per_repo and repo_counts.get(repo_full, 0) >= max_per_repo:
                continue

            workflow_name = (meta.get("name") or meta.get("workflow") or "")
            if max_per_repo_workflow:
                key = (repo_full, workflow_name)
                if repo_workflow_counts.get(key, 0) >= max_per_repo_workflow:
                    continue

            if run_preflight:
                preflight_checked += 1
                if quick_gt_preflight is None:
                    continue
                action = quick_gt_preflight(owner, repo_name, sha, branch, started_at)
                if action not in preflight_actions:
                    continue
                preflight_passed += 1
            else:
                action = "NO_PREFLIGHT"
            if dedupe_repo:
                seen_repos.add(repo_full)

            if not dedupe_repo:
                repo_counts[repo_full] = repo_counts.get(repo_full, 0) + 1
            if max_per_repo_workflow:
                key = (repo_full, workflow_name)
                repo_workflow_counts[key] = repo_workflow_counts.get(key, 0) + 1

            # ── full pipeline for the case ─────────────────────────
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
                        cleaned = ml.strip()[:120]
                        if cleaned and cleaned not in sample_errors:
                            sample_errors.append(cleaned)

                    for w in ex.error_windows:
                        for ln in w[-5:]:
                            ln_clean = ln.strip()
                            if any(
                                kw in ln_clean.lower()
                                for kw in ("error", "failed", "fatal", "not found", "timeout", "exception")
                            ):
                                if ln_clean and ln_clean not in sample_errors and len(sample_errors) < 6:
                                    sample_errors.append(ln_clean[:120])

                cl = None
                if run_classification and classify is not None:
                    if client is None:
                        raise RuntimeError("LLM client was not created")
                    cl = classify(event, client, model=model, log_excerpts=excerpts if excerpts else None)

                gt = None
                if run_ground_truth:
                    if scrape_ground_truth is None:
                        raise RuntimeError("ground truth scraper not available")
                    gt = scrape_ground_truth(event)

                judge_obj = None
                if judge and cl is not None and gt is not None:
                    if client is None:
                        raise RuntimeError("LLM client was not created")
                    judge_obj = llm_judge(build_case_summary(event, sample_errors), asdict(cl), gt, client)

                excerpt_payload = None
                if include_log_excerpts:
                    excerpt_payload = []
                    for ex in excerpts:
                        text = ex.as_prompt_text()
                        if redact:
                            text = _redact_secrets(text)
                        # Enforce size *after* redaction (redaction can slightly expand text).
                        if max_log_chars and len(text) > max_log_chars:
                            text = text[: max_log_chars - 1] + "…"
                        excerpt_payload.append(
                            {
                                "job_file": getattr(ex, "job_file", ""),
                                "step_label": getattr(ex, "step_label", ""),
                                "strategy": getattr(ex, "strategy_used", ""),
                                "text": text,
                            }
                        )

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
                        "total_lines_extracted": sum(sum(len(w) for w in ex.error_windows) for ex in excerpts),
                        "error_markers_found": sum(len(ex.error_marker_lines) for ex in excerpts),
                        "sample_error_lines": sample_errors[:5],
                        "mentioned_files": [asdict(m) for m in cl.mentioned_files] if cl is not None else [],
                    },
                    "classification": (
                        {
                            "category": cl.category,
                            "severity": cl.severity,
                            "confidence": cl.confidence,
                            "action": cl.action,
                            "reasoning": cl.reasoning,
                            "evidence": cl.evidence,
                            "unknowns": cl.unknowns,
                            "mentioned_files": [asdict(m) for m in cl.mentioned_files],
                            "fix_suggestion": asdict(cl.fix_suggestion) if cl.fix_suggestion else None,
                        }
                        if cl is not None
                        else {
                            "category": "UNKNOWN",
                            "severity": "UNKNOWN",
                            "confidence": 0.0,
                            "action": "UNKNOWN",
                            "reasoning": "",
                            "evidence": [],
                            "unknowns": [],
                            "mentioned_files": [],
                            "fix_suggestion": None,
                        }
                    ),
                    "ground_truth": (
                        {
                            "developer_action": gt.developer_action,
                            "method": gt.classification_method,
                            "reasoning": gt.developer_action_reasoning,
                            "signals": gt.signals,
                            "follow_up_commits": [asdict(c) for c in gt.follow_up_commits],
                            "n_follow_up_commits": gt.n_follow_up_commits,
                            "branch_found": gt.branch_found,
                            "repo_accessible": gt.repo_accessible,
                            "api_error": gt.api_error,
                        }
                        if gt is not None
                        else {
                            "developer_action": action if run_preflight else "UNKNOWN",
                            "method": "preflight_only",
                            "reasoning": "",
                            "signals": {},
                            "follow_up_commits": [],
                            "n_follow_up_commits": 0,
                            "branch_found": True,
                            "repo_accessible": True,
                            "api_error": "",
                        }
                    ),
                }

                if excerpt_payload is not None:
                    case["extraction"]["log_excerpts"] = excerpt_payload

                # Keep the same keys demos expect (optional).
                if judge_obj:
                    case["ground_truth"].update(
                        {
                            "match_verdict": judge_obj.get("verdict", "NOT_SCORABLE"),
                            "judge_reasoning": judge_obj.get("reasoning", ""),
                            "judge_confidence": judge_obj.get("confidence", 0.0),
                            "judge_evidence_used": judge_obj.get("evidence_used", []),
                        }
                    )

                if output_format == "json":
                    selected.append(case)
                    writer.write_full_list(selected)
                else:
                    writer.write_case(case)

                selected_count += 1

                gt_action = gt.developer_action if gt is not None else "-"
                print(
                    f"  ✓ [{selected_count:>3}/{n_cases}] {repo_full} "
                    f"branch={branch} preflight={action if run_preflight else 'SKIP'} gt={gt_action}"
                )

            except Exception as e:
                print(f"  ! build case failed for {repo_full}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)

            if selected_count >= n_cases:
                break

    writer.close()

    if run_preflight:
        print(
            f"\nDone. Selected {selected_count} cases. "
            f"Preflight passed {preflight_passed}/{preflight_checked} checks."
        )
    else:
        print(f"\nDone. Selected {selected_count} cases. Preflight skipped.")
    return selected


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract high-quality, scorable CI failure cases into demos/targeted_cases.json format.",
    )

    p.add_argument(
        "--provider",
        type=str,
        default="deepseek",
        choices=("deepseek", "openrouter", "openai", "auto"),
        help="LLM provider to use. Default forces DeepSeek regardless of .env settings.",
    )

    p.add_argument("--runs-path", type=Path, default=Path("/home/guc_alaa/runs.json.gz"))
    p.add_argument("--zip-path", type=Path, default=Path("/home/guc_alaa/github_run_logs.zip"))
    p.add_argument("--out", type=Path, default=Path("/home/guc_alaa/demos/targeted_cases.json"))

    p.add_argument(
        "--format",
        type=str,
        default="json",
        choices=("json", "jsonl", "jsonl.gz"),
        help="Output format. Use jsonl.gz for large datasets intended for git.",
    )
    p.add_argument("--append", action="store_true", help="Append to an existing jsonl/jsonl.gz output")

    p.add_argument("--n", type=int, default=25, help="Number of cases to extract")
    p.add_argument("--scan-start", type=int, default=250_000)
    p.add_argument("--scan-end", type=int, default=520_000)
    p.add_argument("--progress-every", type=int, default=10_000)

    p.add_argument("--no-protected-only", action="store_true", help="Allow non-protected branches")
    p.add_argument("--min-steps-per-job", type=int, default=2)
    p.add_argument("--no-require-shell", action="store_true")
    p.add_argument("--no-dedupe-repo", action="store_true")

    p.add_argument(
        "--preflight-actions",
        type=str,
        default="CODE_FIX,WORKFLOW_FIX,PIN_VERSION,REVERT",
        help="Comma-separated actions to accept from quick_gt_preflight",
    )

    p.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip quick_gt_preflight entirely (fast intake+logs dataset mode).",
    )

    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name. If omitted, chooses a provider-appropriate default.",
    )
    p.add_argument("--no-judge", action="store_true", help="Skip evaluation judge (match_verdict)")

    p.add_argument(
        "--include-log-excerpts",
        action="store_true",
        help="Embed extracted (sanitized) log excerpts per case (can increase output size).",
    )
    p.add_argument(
        "--max-log-chars",
        type=int,
        default=20_000,
        help="Max characters per embedded excerpt text (only if --include-log-excerpts).",
    )
    p.add_argument("--no-redact", action="store_true", help="Disable best-effort secret redaction")

    p.add_argument(
        "--no-classify",
        action="store_true",
        help="Skip LLM classification (keeps schema but uses placeholders).",
    )
    p.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Skip ground-truth scraping (keeps schema but uses preflight-only placeholders).",
    )

    p.add_argument(
        "--max-failed-steps",
        type=int,
        default=3,
        help="Max failed steps per run to extract logs for (use 1 for big datasets).",
    )

    p.add_argument(
        "--max-per-repo",
        type=int,
        default=0,
        help="If >0 and --no-dedupe-repo is set, cap how many cases we take per repo.",
    )
    p.add_argument(
        "--max-per-repo-workflow",
        type=int,
        default=0,
        help="If >0, cap how many cases we take per (repo, workflow) pair.",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    provider = str(args.provider).lower().strip()
    if provider != "auto":
        # Force provider for this run even if other modules load `.env`.
        os.environ["LLM_PROVIDER"] = provider
    else:
        from src.apa.llm_config import get_provider

        provider = get_provider()

    if args.model is None:
        if provider == "deepseek":
            args.model = "deepseek-chat"
        else:
            args.model = "gpt-4.1-mini"

    actions = {a.strip() for a in args.preflight_actions.split(",") if a.strip()}

    extract_cases(
        runs_path=args.runs_path,
        zip_path=args.zip_path,
        out_path=args.out,
        n_cases=int(args.n),
        scan_start=int(args.scan_start),
        scan_end=int(args.scan_end),
        protected_only=not args.no_protected_only,
        min_steps_per_job=int(args.min_steps_per_job),
        require_shell_step=not args.no_require_shell,
        dedupe_repo=not args.no_dedupe_repo,
        preflight_actions=actions,
        progress_every=int(args.progress_every),
        model=str(args.model),
        provider=provider,
        judge=not args.no_judge,
        run_preflight=not bool(args.no_preflight),
        output_format=str(args.format),
        append=bool(args.append),
        include_log_excerpts=bool(args.include_log_excerpts),
        max_log_chars=int(args.max_log_chars),
        redact=not bool(args.no_redact),
        run_classification=not bool(args.no_classify),
        run_ground_truth=not bool(args.no_ground_truth),
        max_failed_steps=int(args.max_failed_steps),
        max_per_repo=int(args.max_per_repo),
        max_per_repo_workflow=int(args.max_per_repo_workflow),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
