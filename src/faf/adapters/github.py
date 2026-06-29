import json
import urllib.request
from datetime import datetime
from typing import List, Optional, Tuple, Any

from faf.adapters.base import FailureAdapter
from faf.models import RunEvent, FailedStepInfo
from faf.exceptions import AdapterAuthenticationError, InsufficientMetadataError

_PROTECTED = ("main", "master", "release", "releases/", "prod", "production")

_TOOLING_ARTIFACT_PATTERNS = (
    "bash-command-extractor",
    "Converting circular structure to JSON",
    "BashWord",
    "Parser exception",
)

def _is_protected(branch: str) -> bool:
    if not branch:
        return False
    b = branch.lower()
    return any(b == p or b.startswith(p) for p in _PROTECTED)

def _describe_step(step: dict) -> Tuple[str, str]:
    step_type = step.get("type", "unknown")
    if step_type == "action":
        repo = step.get("repository", "")
        action = step.get("action", "")
        version = step.get("version", "")
        label = f"{repo}/{action}@{version}".strip("/@") or "action"
    else:
        label = (
            step.get("name")
            or step.get("category")
            or step.get("command")
            or str(step.get("code", ""))[:80]
            or step_type
        )
    return str(label)[:200], step_type

def _extract_error(step: dict) -> Optional[str]:
    for key in ("error", "errors", "error_message", "failure_reason"):
        val = step.get(key)
        if not val:
            continue
        if isinstance(val, str):
            return val[:500]
        if isinstance(val, dict):
            return json.dumps(val)[:500]
        if isinstance(val, list) and val:
            return str(val[0])[:500]
    return None

class GitHubAdapter(FailureAdapter):
    """
    Adapter for GitHub Actions.
    """
    def __init__(self, token: Optional[str] = None):
        self.token = token

    @property
    def source_name(self) -> str:
        return "github"

    @property
    def available_signals(self) -> List[str]:
        return [
            "error_text",
            "jobs_failed",
            "branch_type",
            "commit_message",
            "previous_runs",
            "parent_commit_run",
            "detection_mode"
        ]

    def fetch_event(self, event_id: str, **kwargs: Any) -> RunEvent:
        """
        Fetch the failure details.
        
        Args:
            event_id: The GitHub run ID.
            **kwargs: Must contain 'repo' (e.g., 'owner/repo').
        """
        repo = kwargs.get("repo")
        if not repo:
            raise InsufficientMetadataError("GitHubAdapter requires 'repo' kwarg to fetch an event.")
        return self.from_run_id(repo, event_id)

    def from_run_id(self, repo: str, run_id: str) -> RunEvent:
        """
        Actively fetches a failed GitHub workflow run and its log excerpts using the GitHub REST API.
        """
        if not self.token:
            raise AdapterAuthenticationError("GitHub token is required to fetch from the API.")
            
        # 1. Fetch workflow run metadata
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        
        try:
            with urllib.request.urlopen(req) as response:
                run_data = json.loads(response.read())
        except Exception as e:
            raise InsufficientMetadataError(f"Failed to fetch run {run_id} for repo {repo}: {e}")

        # 2. Fetch jobs
        jobs_url = run_data.get("jobs_url")
        if not jobs_url:
            raise InsufficientMetadataError("No jobs URL found in the run metadata.")
            
        req_jobs = urllib.request.Request(jobs_url)
        req_jobs.add_header("Authorization", f"Bearer {self.token}")
        req_jobs.add_header("Accept", "application/vnd.github.v3+json")
        
        try:
            with urllib.request.urlopen(req_jobs) as response:
                jobs_data = json.loads(response.read())
        except Exception as e:
            raise InsufficientMetadataError(f"Failed to fetch jobs for run {run_id}: {e}")

        # 3. Build a pseudo raw_run dict to reuse the robust parse logic
        raw_run = {
            "_id": run_id,
            "repository_name": repo,
            "workflow_path": run_data.get("path", ""),
            "run_number": run_data.get("run_number", 0),
            "run_attempt": run_data.get("run_attempt", 0),
            "metadata": {
                "event": run_data.get("event", ""),
                "head_branch": run_data.get("head_branch", ""),
                "conclusion": run_data.get("conclusion", "unknown"),
                "head_commit": run_data.get("head_commit", {}),
                "actor": run_data.get("actor", {}),
                "run_started_at": run_data.get("run_started_at", ""),
                "updated_at": run_data.get("updated_at", "")
            },
            "log_insights": []
        }

        # Format jobs to roughly match the expected log_insights structure
        for job in jobs_data.get("jobs", []):
            job_insight = {
                "file": job.get("name", ""),
                "duration_sec": 0, # not perfectly computed here but acceptable
                "steps": []
            }
            for step in job.get("steps", []):
                formatted_step = {
                    "name": step.get("name", ""),
                    "status": step.get("status", ""),
                    "conclusion": step.get("conclusion", ""),
                    # If failed, we treat the name as a weak error proxy since we don't 
                    # download the full logs in this basic API implementation.
                    # A robust implementation would fetch the log url and parse it.
                }
                if step.get("conclusion") == "failure":
                    formatted_step["error"] = f"Step '{step.get('name')}' failed."
                job_insight["steps"].append(formatted_step)
            
            raw_run["log_insights"].append(job_insight)

        return self.parse_raw(raw_run)

    def parse_raw(self, raw_run: dict) -> RunEvent:
        """
        Parses a raw run JSON dict (e.g. from the thesis evaluation corpus).
        """
        meta = raw_run.get("metadata", {})
        head_commit = meta.get("head_commit", {})
        commit_msg = head_commit.get("message", "")
        branch = meta.get("head_branch", "")
        conclusion = meta.get("conclusion", "unknown")
        log_insights = raw_run.get("log_insights", [])

        failed_steps: List[FailedStepInfo] = []
        failed_jobs_count = 0
        run_detection_mode = "unknown_failure"

        if conclusion == "failure" and log_insights:
            per_job_modes: List[str] = []
            for job in log_insights:
                steps = job.get("steps") or []
                idx = -1
                mode = "unknown_failure"
                
                # find first failing step
                for i, step in enumerate(steps):
                    if _extract_error(step):
                        idx = i
                        mode = "per_step_error"
                        break
                        
                if idx == -1 and steps:
                    idx = len(steps) - 1
                    mode = "job_level_fallback"
                    
                per_job_modes.append(mode)

                if idx == -1:
                    continue

                failed_jobs_count += 1
                step = steps[idx] if idx < len(steps) else {}
                label, step_type = _describe_step(step)
                error_text = _extract_error(step) or ""

                failed_steps.append(
                    FailedStepInfo(
                        name=label,
                        error_text=error_text,
                        duration_seconds=step.get("duration_sec"),
                        metadata={
                            "job_file": job.get("file", ""),
                            "runner_image": job.get("image", ""),
                            "step_type": step_type,
                            "detection_mode": mode,
                            "tooling_artifact_suspected": any(p in error_text for p in _TOOLING_ARTIFACT_PATTERNS)
                        }
                    )
                )

            weakness_order = {
                "per_step_error": 0,
                "single_step_inferred": 1,
                "job_level_fallback": 2,
                "unknown_failure": 3,
            }
            if failed_jobs_count > 0:
                attributed_modes = [m for m in per_job_modes if m != "unknown_failure"] or per_job_modes
                run_detection_mode = max(attributed_modes, key=lambda m: weakness_order.get(m, 99))

        return RunEvent(
            source=self.source_name,
            run_id=str(raw_run.get("_id", "")),
            status=conclusion,
            url=f"https://github.com/{raw_run.get('repository_name', '')}/actions/runs/{raw_run.get('_id', '')}",
            failed_steps=failed_steps,
            available_signals=self.available_signals,
            metadata={
                "repo": raw_run.get("repository_name", ""),
                "workflow": raw_run.get("workflow_path", ""),
                "branch": branch,
                "is_protected_branch": _is_protected(branch),
                "commit_sha": str(head_commit.get("id", ""))[:12],
                "commit_message": commit_msg,
                "jobs_total": len(log_insights),
                "jobs_failed": failed_jobs_count,
                "detection_mode": run_detection_mode,
            }
        )
