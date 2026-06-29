# ground_truth_scraper.py
# ─────────────────────────────────────────────────────────────────────
# Given a failed run, determine what the developer actually did next.
# This is the ground truth proxy for thesis evaluation.
#
# Strategy (in order of reliability):
#   1. Check for follow-up commits on the same branch.
#      → Apply deterministic rubric to classify.
#      → If rubric is ambiguous, use LLM to interpret.
#   2. If branch is deleted, search for the PR by commit SHA.
#      → Check if PR was merged or closed.
#   3. If no PR found, check for workflow re-runs.
#      → A successful re-run = developer chose RETRY.
#   4. If nothing found → UNKNOWN.
#
# The rubric is pre-registered: the thesis methodology chapter should
# describe it before presenting evaluation results.
# ─────────────────────────────────────────────────────────────────────


    #     CI FAILURE EVENT                                                   
    #             │
    #             ▼
    #  ┌─────────────────────┐
    #  │ Follow-up commits?  │
    #  └─────────┬───────────┘
    #            │ YES
    #            ▼
    #  ┌─────────────────────┐
    #  │ Apply rubric rules  │
    #  └─────────┬───────────┘
    #            │
    #   ┌────────┴─────────┐
    #   │ Confident?       │
    #   └───────┬──────────┘
    #           │ YES                NO
    #           ▼                    ▼
    #  ┌───────────────┐     ┌──────────────────┐
    #  │ Final label   │     │ LLM classification│
    #  └───────────────┘     └──────────────────┘


    #            │ NO commits
    #            ▼
    #  ┌─────────────────────┐
    #  │ PR exists?          │
    #  └─────────┬───────────┘
    #            │ YES
    #            ▼
    #  ┌─────────────────────┐
    #  │ Check PR state      │
    #  └─────────┬───────────┘
    #            ▼
    #     PR_MERGED / CLOSED / OPEN


    #            │ NO PR
    #            ▼
    #  ┌─────────────────────┐
    #  │ Successful rerun?   │
    #  └─────────┬───────────┘
    #            │ YES
    #            ▼
    #     RETRY_SUCCEEDED


    #            │ NO
    #            ▼
    #         UNKNOWN





#     Follow-up commits
#         │
#         ▼
# Extract signals (keywords + files)
#         │
#         ▼
# ┌──────────────────────────────┐
# │ has_revert? → REVERT         │
# ├──────────────────────────────┤
# │ has_pin + deps? → PIN        │
# ├──────────────────────────────┤
# │ workflow only? → WF_FIX      │
# ├──────────────────────────────┤
# │ fix + code? → CODE_FIX       │
# └──────────────────────────────┘
#         │
#         ▼
# Same input → SAME output every time

import os
import json
import time
import requests
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dotenv import load_dotenv
from src.apa.llm_usage import record_usage

load_dotenv()

# Fix-attribution window (days): a follow-up commit only counts as the developer's
# remediation if it lands within this many days of the failure. Default 7 (matches the
# original corpus). Topping-up may widen it (e.g. GT_FIX_WINDOW_DAYS=14) to capture
# delayed fixes; any case mined under a wider window must be disclosed in the writeup.
import os as _os
GT_FIX_WINDOW_DAYS = int(_os.environ.get("GT_FIX_WINDOW_DAYS", "7"))


# ─── typed output schema ─────────────────────────────────────────────

@dataclass
class FollowUpCommit:
    sha: str
    message_title: str
    message_full: str
    author: str
    date: str
    files_changed: List[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    file_summaries: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class GroundTruth:
    run_id: str
    repo: str
    branch: str
    failing_sha: str

    n_follow_up_commits: int = 0
    follow_up_commits: List[FollowUpCommit] = field(default_factory=list)

    developer_action: str = "UNKNOWN"
    developer_action_reasoning: str = ""
    classification_method: str = "none"  # rubric | llm | pr_state | rerun | none

    signals: Dict[str, object] = field(default_factory=dict)

    api_error: str = ""
    branch_found: bool = False
    repo_accessible: bool = False


# ─── GitHub API helpers ──────────────────────────────────────────────

def _get_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _api_get(url: str, params: dict = None) -> Optional[object]:
    headers = _get_headers()
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 403:
            reset_time = resp.headers.get("X-RateLimit-Reset")
            if reset_time:
                wait = int(reset_time) - int(time.time()) + 1
                if 0 < wait < 120:
                    print(f"    rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code == 422:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    API error: {e}")
        return None


# ─── GitHub API: fetch commits ───────────────────────────────────────

def fetch_commits_after(
    owner: str, repo: str, branch: str,
    since_sha: str, since_date: str,
    max_commits: int = 5,
) -> Optional[List[dict]]:
    # Primary: compare API for accurate chronological ordering
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{since_sha}...{branch}"
    data = _api_get(url)
    if data and isinstance(data, dict):
        commits = data.get("commits") or []
        return commits[:max_commits]

    # Fallback: since-based approach
    try:
        dt = datetime.fromisoformat(since_date.replace("Z", "+00:00"))
        since_iso = (dt + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        since_iso = since_date

    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"sha": branch, "since": since_iso, "per_page": 100}
    data = _api_get(url, params)
    if data is None or not isinstance(data, list):
        return None

    filtered = [c for c in data if c.get("sha") != since_sha]
    filtered.sort(
        key=lambda c: (c.get("commit") or {}).get("committer", {}).get("date", "")
    )
    return filtered[:max_commits]


def fetch_commit_stats(owner: str, repo: str, sha: str) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    data = _api_get(url)
    if data is None:
        return {"additions": 0, "deletions": 0, "files": []}
    stats = data.get("stats") or {}
    files = [
        {
            "filename": f.get("filename", ""),
            "status": f.get("status", ""),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
            "patch_excerpt": (f.get("patch") or "")[:1200],
        }
        for f in (data.get("files") or [])
    ]
    return {
        "additions": stats.get("additions", 0),
        "deletions": stats.get("deletions", 0),
        "files": files,
        "author": ((data.get("commit") or {}).get("author") or {}).get("name", ""),
    }


# ─── GitHub API: find PR by commit SHA ───────────────────────────────

def find_pr_by_commit_sha(owner: str, repo: str, sha: str) -> Optional[dict]:
    """Find PRs associated with a specific commit SHA."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/pulls"
    headers = _get_headers()
    headers["Accept"] = "application/vnd.github.groot-preview+json"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        prs = resp.json()
        if prs and isinstance(prs, list) and len(prs) > 0:
            return prs[0]
        return None
    except requests.RequestException:
        return None


def find_pr_by_branch(owner: str, repo: str, branch: str) -> Optional[dict]:
    """Search for a PR opened from the given branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {"state": "all", "head": f"{owner}:{branch}", "per_page": 5}
    data = _api_get(url, params)
    if data and isinstance(data, list) and len(data) > 0:
        return data[0]
    return None


def fetch_pr_files(owner: str, repo: str, pr_number: int) -> List[dict]:
    """Fetch the list of files changed in a merged PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    data = _api_get(url, params={"per_page": 100})
    if not data or not isinstance(data, list):
        return []
    return data


def classify_pr_fix_type(pr_data: dict, pr_files: List[dict]) -> tuple:
    """
    Classify what type of fix a merged PR actually contained,
    using the same rubric as classify_developer_action_rubric.
    Returns (action, reasoning, signals).
    """
    pr_title = pr_data.get("title", "") or ""
    pr_body = pr_data.get("body", "") or ""
    msg = f"{pr_title}\n{pr_body}"
    files_changed = [f.get("filename", "") for f in pr_files]

    signals = {
        "has_revert": _check_message(msg, REVERT_KEYWORDS),
        "has_pin": _check_message(msg, PIN_KEYWORDS),
        "has_fix": _check_message(msg, FIX_KEYWORDS),
        "has_retry": _check_message(msg, RETRY_KEYWORDS),
        "changed_workflow": _check_files(files_changed, WORKFLOW_PATTERNS),
        "changed_dependencies": _check_files(files_changed, DEPENDENCY_PATTERNS),
        "changed_config": _check_files(files_changed, CONFIG_PATTERNS),
        "changed_source_code": False,
        "n_files": len(files_changed),
    }

    source_files = [
        f for f in files_changed
        if not any(pat in f.lower() for pat in
                   WORKFLOW_PATTERNS + DEPENDENCY_PATTERNS + CONFIG_PATTERNS)
    ]
    if source_files:
        signals["changed_source_code"] = True

    short_files = ", ".join(files_changed[:8])

    if signals["has_revert"]:
        return "REVERT", f"PR reverted a change. Title: {pr_title[:100]}", signals
    if signals["has_pin"] and signals["changed_dependencies"]:
        return "PIN_VERSION", f"PR pinned/changed a dependency. Files: {short_files}", signals
    if signals["changed_workflow"] and not signals["changed_source_code"]:
        return "WORKFLOW_FIX", f"PR changed only CI config. Files: {short_files}", signals
    if signals["has_fix"] and signals["changed_source_code"] and not signals["changed_workflow"]:
        return "CODE_FIX", f"PR fixed source code. Files: {short_files}", signals
    if signals["changed_source_code"]:
        return "CODE_CHANGE", f"PR changed source code (fix keyword absent). Files: {short_files}", signals
    if signals["changed_dependencies"]:
        return "DEPENDENCY_CHANGE", f"PR changed dependencies. Files: {short_files}", signals
    if not files_changed:
        return "PR_MERGED_NO_FILES", "PR was merged but no file details could be fetched.", signals

    return "PR_MERGED_UNCLEAR", f"PR merged, change type unclear. Files: {short_files}", signals


# ─── GitHub API: check for workflow re-runs ──────────────────────────

def check_for_successful_rerun(
    owner: str, repo: str, branch: str,
    workflow_path: str, after_date: str,
    window_hours: int = 48,
) -> Optional[dict]:
    """
    Check if the same workflow was re-run on the same branch shortly
    after the failure, and if so, whether it succeeded.
    """
    try:
        dt = datetime.fromisoformat(after_date.replace("Z", "+00:00"))
        created_after = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        created_before = (dt + timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    params = {
        "branch": branch,
        "created": f"{created_after}..{created_before}",
        "per_page": 10,
    }
    data = _api_get(url, params)
    if not data or not isinstance(data, dict):
        return None

    runs = data.get("workflow_runs") or []
    for run in runs:
        if run.get("conclusion") == "success":
            run_path = run.get("path", "")
            if workflow_path in run_path or run_path in workflow_path:
                return {
                    "run_id": run.get("id"),
                    "conclusion": "success",
                    "created_at": run.get("created_at"),
                    "html_url": run.get("html_url"),
                }
    return None


# ─── deterministic rubric ────────────────────────────────────────────

WORKFLOW_PATTERNS = (".github/workflows/", ".github/actions/")
DEPENDENCY_PATTERNS = (
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "requirements.txt", "Pipfile", "Pipfile.lock", "poetry.lock",
    "pyproject.toml", "setup.py", "setup.cfg",
    "Cargo.toml", "Cargo.lock",
    "pom.xml", "build.gradle", "build.gradle.kts",
    "go.mod", "go.sum",
    "Gemfile", "Gemfile.lock",
    "composer.json", "composer.lock",
)
CONFIG_PATTERNS = (
    ".github/", "Dockerfile", "docker-compose",
    ".ci/", "ci/", "Makefile", "CMakeLists",
    "tox.ini", ".eslintrc", ".prettierrc",
)

REVERT_KEYWORDS = ("revert ", "reverted ", "revert:", "undo ", "rollback ", "rolling back")
PIN_KEYWORDS = ("pin ", "pinned ", "lock ", "freeze ", "downgrade ", "use v", "use version", "bump ")
FIX_KEYWORDS = ("fix ", "fix:", "fixed ", "hotfix", "patch ", "repair ", "resolve ", "workaround")
RETRY_KEYWORDS = ("retry", "re-run", "rerun", "trigger", "rebuild")


def _check_message(msg: str, keywords: tuple) -> bool:
    return any(kw in msg.lower() for kw in keywords)


def _check_files(files: List[str], patterns: tuple) -> bool:
    return any(any(pat in f.lower() for pat in patterns) for f in files)


# ── Relevance filter ─────────────────────────────────────────────────
# A follow-up commit is only credible as "the fix" if it is plausibly a
# RESPONSE to this failure, not just the next thing that happened on the
# branch. Temporal proximity alone produces false positives (an unrelated
# feature commit, a merge, someone else's parallel work). A commit is
# relevant only on CONCRETE evidence, not keyword heuristics:
#   - same author as the failing commit (the person who broke it fixed it)
#   - it touches a file that the failing commit also touched (file overlap)
# A message-keyword signal was considered and rejected: a CI/fix-sounding
# message can equally come from someone doing unrelated CI work, so it is
# not concrete enough to attribute the commit to this failure.


def _commit_is_relevant(
    commit: "FollowUpCommit",
    failing_author: str,
    failing_files: set,
) -> tuple[bool, str]:
    """Return (is_relevant, reason). Relevance requires concrete overlap."""
    # 1. same author as the failing commit
    if failing_author and commit.author and \
            commit.author.strip().lower() == failing_author.strip().lower():
        return True, "same_author"
    # 2. file overlap with the failing commit
    if failing_files:
        cf = {f.lower() for f in (commit.files_changed or [])}
        if cf & failing_files:
            return True, "file_overlap"
    return False, "unrelated"


def classify_developer_action_rubric(
    follow_ups: List[FollowUpCommit],
    failure_date: str = "",
    failing_author: str = "",
    failing_files: set = None,
) -> tuple:
    """
    Apply deterministic rubric. Returns (action, reasoning, signals, confident).
    The `confident` flag indicates whether the rubric is sure enough
    to skip the LLM fallback.

    Follow-ups are first passed through a relevance filter: a commit counts
    only if it is timely AND plausibly a response to the failure (same author,
    file overlap with the failing commit, or a CI/fix-referencing message).
    Timely-but-unrelated commits are dropped, which prevents attributing the
    next unrelated commit on the branch to this failure.
    """
    failing_files = failing_files or set()
    if not follow_ups:
        return ("NO_FOLLOW_UP",
                "No follow-up commits found.",
                {"no_commits": True}, True)

    signals = {
        "has_revert": False, "has_pin": False, "has_fix": False,
        "has_retry": False, "changed_workflow": False,
        "changed_dependencies": False, "changed_config": False,
        "changed_source_code": False, "commits_within_7_days": 0,
        "relevant_commits": 0, "relevance_reasons": [],
    }

    all_messages = []
    all_files = []

    # Check time proximity — commits more than 7 days after the
    # failure are probably unrelated
    try:
        failure_dt = datetime.fromisoformat(failure_date.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        failure_dt = None

    relevant_commits = []
    for commit in follow_ups:
        # Check if this commit is within 7 days of the failure
        is_timely = True
        if failure_dt and commit.date:
            try:
                commit_dt = datetime.fromisoformat(commit.date.replace("Z", "+00:00"))
                days_after = (commit_dt - failure_dt).days
                if days_after <= GT_FIX_WINDOW_DAYS:
                    signals["commits_within_7_days"] += 1
                else:
                    is_timely = False
            except (ValueError, TypeError):
                pass

        # Relevance filter: a timely commit only counts toward the fix label
        # if it is plausibly a response to THIS failure.
        relevant, reason = _commit_is_relevant(commit, failing_author, failing_files)
        if not (is_timely and relevant):
            continue

        signals["relevant_commits"] += 1
        signals["relevance_reasons"].append(reason)
        relevant_commits.append(commit)
        all_messages.append(commit.message_full)
        all_files.extend(commit.files_changed)

        msg = commit.message_full
        files = commit.files_changed

        if _check_message(msg, REVERT_KEYWORDS):
            signals["has_revert"] = True
        if _check_message(msg, PIN_KEYWORDS):
            signals["has_pin"] = True
        if _check_message(msg, FIX_KEYWORDS):
            signals["has_fix"] = True
        if _check_message(msg, RETRY_KEYWORDS):
            signals["has_retry"] = True
        if _check_files(files, WORKFLOW_PATTERNS):
            signals["changed_workflow"] = True
        if _check_files(files, DEPENDENCY_PATTERNS):
            signals["changed_dependencies"] = True
        if _check_files(files, CONFIG_PATTERNS):
            signals["changed_config"] = True

        source_files = [
            f for f in files
            if not any(pat in f.lower() for pat in
                       WORKFLOW_PATTERNS + DEPENDENCY_PATTERNS + CONFIG_PATTERNS)
        ]
        if source_files:
            signals["changed_source_code"] = True

    # If no commits are within 7 days, the follow-ups are probably
    # unrelated to the failure
    if signals["commits_within_7_days"] == 0 and failure_dt:
        return (
            "NO_IMMEDIATE_RESPONSE",
            "Follow-up commits exist but none are within 7 days of the "
            "failure. Developer likely retried or ignored the failure "
            "without a code change.",
            signals, True,
        )

    # If timely commits exist but NONE are relevant to this failure (different
    # author, no file overlap, no CI/fix reference), we cannot credibly call
    # any of them "the fix". Mark not-scorable rather than guess.
    if signals["relevant_commits"] == 0:
        return (
            "NO_RELEVANT_RESPONSE",
            "Follow-up commits exist within the window but none are plausibly "
            "a response to this failure (different author, no overlap with the "
            "failing commit's files, and no CI/fix reference in the message).",
            signals, True,
        )

    # Apply rubric in priority order (high confidence cases)
    if signals["has_revert"]:
        return ("REVERT",
                f"Developer reverted. Messages: "
                f"{'; '.join(m.splitlines()[0] for m in all_messages[:3])}",
                signals, True)

    if signals["has_pin"] and signals["changed_dependencies"]:
        return ("PIN_VERSION",
                f"Developer pinned dependencies. "
                f"Files: {', '.join(list(set(all_files))[:10])}",
                signals, True)

    if signals["changed_workflow"] and not signals["changed_source_code"]:
        return ("WORKFLOW_FIX",
                f"Developer edited CI config only. "
                f"Files: {', '.join(list(set(all_files))[:10])}",
                signals, True)

    if signals["has_fix"] and signals["changed_source_code"] and not signals["changed_workflow"]:
        return ("CODE_FIX",
                f"Developer fixed source code. "
                f"Files: {', '.join(list(set(all_files))[:10])}",
                signals, True)

    # Ambiguous cases — not confident enough, use LLM
    return ("AMBIGUOUS",
            "Multiple signals present, rubric cannot determine a single action.",
            signals, False)


# ─── LLM fallback for ambiguous cases ────────────────────────────────

def classify_with_llm(
    event_summary: str,
    follow_ups: List[FollowUpCommit],
) -> tuple:
    """
    Use OpenAI chat completions to interpret ambiguous follow-up commits.
    Returns (action, reasoning).
    """
    from src.apa.llm_config import make_client, provider_env_name

    try:
        client = make_client()
    except RuntimeError:
        return ("EVALUATION_ERROR", f"No {provider_env_name()} available for LLM fallback.")

    # Tiny follow-up commit format: keep only title and top 5 filenames
    commits_text = ""
    for i, c in enumerate(follow_ups[:2], 1):
        files = getattr(c, "files_changed", []) or []
        files_str = ", ".join(files[:5])
        commits_text += (
            f"\nCommit {i}: {getattr(c, 'sha', '')}\n"
            f"title: {getattr(c, 'message_title', '')[:150]}\n"
            f"files: {files_str}\n"
        )

    prompt = f"""You are analyzing what a developer did after a CI/CD failure.

ORIGINAL FAILURE:
{event_summary}

FOLLOW-UP COMMITS (in chronological order after the failure):
{commits_text}

Based on these follow-up commits, classify the developer's response to the CI failure.
Choose exactly ONE action from this list:

  REVERT — Developer reverted or undid the failing change.
  PIN_VERSION — Developer pinned, downgraded, or changed a dependency/tool version.
  WORKFLOW_FIX — Developer edited CI/workflow configuration to fix the build.
  CODE_FIX — Developer fixed the source code that caused the failure.
  CODE_CHANGE — Developer changed source code but the change may not be directly related to the failure.
  RETRY — Developer simply retried without meaningful code changes.
  UNRELATED — The follow-up commits appear unrelated to the CI failure.
  UNKNOWN — Cannot determine from the available information.

Respond with ONLY a JSON object:
{{
  "action": "one of the actions above",
  "reasoning": "2-3 sentences explaining why you chose this action, referencing specific commits"
}}"""

    # Retry with exponential backoff.
    from src.apa.llm_config import get_provider
    provider = get_provider()
    import os
    if provider == "deepseek":
        primary_model = os.environ.get("CI_AGENT_MODEL", "deepseek-reasoner")
    elif provider == "openrouter":
        primary_model = os.environ.get("CI_AGENT_MODEL", "deepseek/deepseek-r1")
    elif provider == "gemini":
        primary_model = os.environ.get("CI_AGENT_MODEL", "gemini-2.5-flash")
    else:
        primary_model = "gpt-4o-mini"
        
    max_attempts = 3
    backoff = 1
    
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        model = primary_model
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You classify developer responses to CI failures. Be precise and cite specific commits."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=250,
            )
            record_usage(response, model, call_type="chat", label="ground_truth_scraper.classify_with_llm")
            content = response.choices[0].message.content or ""
            # Strip DeepSeek <think>...</think> reasoning blocks
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'```(?:json)?', '', content).strip()
            if not content:
                if attempt < max_attempts:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return ("EVALUATION_ERROR", "LLM fallback returned empty response")
            data = json.loads(content)
            return (
                data.get("action", "UNKNOWN"),
                data.get("reasoning", ""),
            )
        except Exception as e:
            last_exc = e
            msg = str(e)
            if "402" in msg or "max_tokens" in msg or "credits" in msg:
                break
            if attempt < max_attempts:
                try:
                    time.sleep(backoff)
                except Exception:
                    pass
                backoff *= 2

    return ("EVALUATION_ERROR", f"LLM fallback failed: {last_exc}")


# ─── cheap pre-flight check (no LLM, no per-commit stats API) ────────

def quick_gt_preflight(owner: str, repo_name: str, sha: str, branch: str, started_at: str) -> str:
    """
    One GitHub API call: fetch commits after sha on branch.
    Apply message-only rubric (no file-stats calls).
    Returns a developer_action string, or 'UNKNOWN'/'NO_IMMEDIATE_RESPONSE'.
    Strong-signal actions: CODE_FIX, WORKFLOW_FIX, PIN_VERSION, REVERT, DEPENDENCY_CHANGE.
    Weak-signal/ambiguous actions: AMBIGUOUS.
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}/compare/{sha}...{branch}"
    data = _api_get(url)
    if not data or not isinstance(data, dict):
        # Fallback: try PR lookup by SHA
        pr = find_pr_by_commit_sha(owner, repo_name, sha)
        if pr and pr.get("merged_at"):
            return "PR_MERGED"
        return "UNKNOWN"

    commits = data.get("commits") or []
    if not commits:
        return "NO_IMMEDIATE_RESPONSE"

    try:
        failure_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        failure_dt = None

    timely = []
    for c in commits[:10]:
        commit_data = c.get("commit") or {}
        date = (commit_data.get("committer") or {}).get("date", "")
        if failure_dt and date:
            try:
                cdt = datetime.fromisoformat(date.replace("Z", "+00:00"))
                if (cdt - failure_dt).days > GT_FIX_WINDOW_DAYS:
                    continue
            except (ValueError, TypeError):
                pass
        timely.append((commit_data.get("message", ""), []))

    if not timely:
        return "NO_IMMEDIATE_RESPONSE"

    # Message-only rubric (no file signals — just keywords)
    has_revert = any(_check_message(m, REVERT_KEYWORDS) for m, _ in timely)
    has_fix = any(_check_message(m, FIX_KEYWORDS) for m, _ in timely)
    has_pin = any(_check_message(m, PIN_KEYWORDS) for m, _ in timely)

    if has_revert:
        return "REVERT"
    if has_pin:
        return "PIN_VERSION"
    if has_fix:
        return "CODE_FIX"
    # There are timely commits but no strong keyword signal — too ambiguous for a
    # controlled evaluation set. Let the selector exclude these.
    return "AMBIGUOUS"


# ─── the scraper (main entry point) ─────────────────────────────────

def scrape_ground_truth(event) -> GroundTruth:
    result = GroundTruth(
        run_id=event.run_id,
        repo=event.repo,
        branch=event.branch,
        failing_sha=event.commit_sha,
    )

    parts = event.repo.split("/")
    if len(parts) != 2:
        result.api_error = f"Cannot parse owner/repo from '{event.repo}'"
        return result
    owner, repo_name = parts

    # Check repo accessibility
    print(f"  checking repo {owner}/{repo_name}...")
    repo_data = _api_get(f"https://api.github.com/repos/{owner}/{repo_name}")
    if repo_data is None:
        result.api_error = "Repository not found or not accessible"
        return result
    result.repo_accessible = True

    # ── Strategy 1: follow-up commits on the same branch ────────────
    print(f"  fetching commits after {event.commit_sha} on '{event.branch}'...")
    commits = fetch_commits_after(
        owner, repo_name, event.branch,
        event.commit_sha, event.started_at,
    )

    if commits is not None and len(commits) > 0:
        result.branch_found = True
        print(f"  found {len(commits)} follow-up commits, fetching details...")

        for c in commits:
            commit_data = c.get("commit") or {}
            sha = c.get("sha", "")
            msg = commit_data.get("message", "")
            author = (commit_data.get("author") or {}).get("name", "")
            date = (commit_data.get("committer") or {}).get("date", "")
            stats = fetch_commit_stats(owner, repo_name, sha)

            result.follow_up_commits.append(FollowUpCommit(
                sha=sha[:12],
                message_title=msg.splitlines()[0] if msg else "",
                message_full=msg,
                author=author,
                date=date,
                files_changed=[f["filename"] for f in stats.get("files", [])],
                additions=stats.get("additions", 0),
                deletions=stats.get("deletions", 0),
                file_summaries=stats.get("files", []),
            ))

        result.n_follow_up_commits = len(result.follow_up_commits)

        # Relevance inputs: the failing commit's author and the files it
        # changed. The dataset carries neither, so one API call fetches both
        # (author falls back to the event field if the call returns nothing).
        failing_author = getattr(event, "commit_author", "") or ""
        failing_files: set = set()
        if event.commit_sha:
            fstats = fetch_commit_stats(owner, repo_name, event.commit_sha)
            failing_files = {
                f["filename"].lower() for f in fstats.get("files", []) if f.get("filename")
            }
            failing_author = fstats.get("author") or failing_author

        # Apply rubric (with relevance filtering)
        action, reasoning, signals, confident = classify_developer_action_rubric(
            result.follow_up_commits,
            failure_date=event.started_at,
            failing_author=failing_author,
            failing_files=failing_files,
        )
        result.signals = signals

        if confident:
            result.developer_action = action
            result.developer_action_reasoning = reasoning
            result.classification_method = "rubric"
        else:
            # LLM fallback for ambiguous cases
            print(f"  rubric ambiguous, using LLM fallback...")
            event_summary = (
                f"Repo: {event.repo}, Branch: {event.branch}\n"
                f"Commit: {event.commit_sha} — {event.commit_title}\n"
                f"Failure: {event.conclusion}, "
                f"{event.failed_jobs_count} failed jobs\n"
                f"Failure detection: {event.failure_detection}"
            )
            llm_action, llm_reasoning = classify_with_llm(
                event_summary, result.follow_up_commits,
            )
            result.developer_action = llm_action
            result.developer_action_reasoning = llm_reasoning
            result.classification_method = "llm"

        return result

    # ── Strategy 2: PR lookup (branch is gone) ──────────────────────
    result.branch_found = False

    # Try PR by commit SHA first (more reliable)
    print(f"  branch not found, searching PR by commit SHA...")
    pr_data = find_pr_by_commit_sha(owner, repo_name, event.commit_sha)

    # Fall back to PR by branch name
    if pr_data is None:
        print(f"  no PR by SHA, trying branch name...")
        pr_data = find_pr_by_branch(owner, repo_name, event.branch)

    if pr_data is not None:
        pr_state = pr_data.get("state", "unknown")
        pr_merged = pr_data.get("merged_at")
        pr_closed = pr_data.get("closed_at")
        pr_number = pr_data.get("number")
        pr_title = pr_data.get("title", "")

        if pr_merged:
            # Fetch PR files to classify what type of fix was actually made.
            print(f"  PR #{pr_number} merged, fetching files to classify fix type...")
            pr_files = fetch_pr_files(owner, repo_name, pr_number)
            if pr_files:
                action, reasoning, fix_signals = classify_pr_fix_type(pr_data, pr_files)
                result.developer_action = action
                result.developer_action_reasoning = (
                    f"PR #{pr_number} ('{pr_title}') merged on {pr_merged}. {reasoning}"
                )
                result.signals = {
                    **fix_signals,
                    "pr_number": pr_number, "pr_state": pr_state,
                    "pr_merged": True, "source": "pr_files",
                }
                result.classification_method = "pr_files"
            else:
                # No file data — fall back to the opaque PR_MERGED label
                result.developer_action = "PR_MERGED"
                result.developer_action_reasoning = (
                    f"PR #{pr_number} ('{pr_title}') was merged on {pr_merged}. "
                    "No file details available."
                )
                result.signals = {
                    "pr_number": pr_number, "pr_state": pr_state,
                    "pr_merged": True, "source": "pr_lookup",
                }
                result.classification_method = "pr_state"
        elif pr_state == "closed":
            result.developer_action = "PR_CLOSED_UNMERGED"
            result.developer_action_reasoning = (
                f"PR #{pr_number} ('{pr_title}') was closed without "
                f"merging on {pr_closed}. Developer rejected the change."
            )
            result.signals = {
                "pr_number": pr_number, "pr_state": pr_state,
                "pr_merged": False, "pr_closed_unmerged": True,
                "source": "pr_lookup",
            }
            result.classification_method = "pr_state"
        elif pr_state == "open":
            result.developer_action = "PR_STILL_OPEN"
            result.developer_action_reasoning = (
                f"PR #{pr_number} ('{pr_title}') is still open."
            )
            result.signals = {
                "pr_number": pr_number, "pr_state": "open",
                "pr_merged": False, "source": "pr_lookup",
            }
            result.classification_method = "pr_state"
        else:
            result.developer_action = "UNKNOWN"
            result.developer_action_reasoning = f"PR #{pr_number} state unclear: {pr_state}"
            result.signals = {
                "pr_number": pr_number, "pr_state": pr_state,
                "source": "pr_lookup",
            }
            result.classification_method = "pr_state"

        return result

    # ── Strategy 3: check for successful re-run ─────────────────────
    print(f"  no PR found, checking for workflow re-runs...")
    workflow_path = getattr(event, "workflow", "")
    rerun = check_for_successful_rerun(
        owner, repo_name, event.branch,
        workflow_path, event.started_at,
    )

    if rerun:
        result.developer_action = "RETRY_SUCCEEDED"
        result.developer_action_reasoning = (
            f"Workflow was re-run and succeeded on {rerun.get('created_at', '?')}."
        )
        result.classification_method = "rerun"
        result.signals = {"rerun": rerun, "source": "actions_api"}
        return result

    # ── Nothing found ───────────────────────────────────────────────
    result.developer_action = "UNKNOWN"
    result.developer_action_reasoning = (
        "Branch deleted, no PR found by SHA or branch name, "
        "no successful re-run detected."
    )
    result.classification_method = "none"
    return result


# ─── pretty printer ──────────────────────────────────────────────────

def print_ground_truth(gt: GroundTruth) -> None:
    print(f"\n  GROUND TRUTH for {gt.run_id}")
    print(f"    repo:           {gt.repo}")
    print(f"    branch:         {gt.branch}")
    print(f"    failing sha:    {gt.failing_sha}")
    print(f"    repo accessible: {gt.repo_accessible}")
    print(f"    branch found:   {gt.branch_found}")
    print(f"    method:         {gt.classification_method}")
    if gt.api_error:
        print(f"    API error:      {gt.api_error}")
    print(f"    follow-ups:     {gt.n_follow_up_commits}")

    for i, c in enumerate(gt.follow_up_commits, 1):
        print(f"\n    commit [{i}]  {c.sha}")
        print(f"      date:    {c.date}")
        print(f"      author:  {c.author}")
        print(f"      message: {c.message_title}")
        print(f"      files:   {', '.join(c.files_changed[:8])}")
        if len(c.files_changed) > 8:
            print(f"               ... and {len(c.files_changed) - 8} more")
        print(f"      +{c.additions} -{c.deletions}")

    print(f"\n    DEVELOPER ACTION: {gt.developer_action}")
    print(f"    method:    {gt.classification_method}")
    print(f"    reasoning: {gt.developer_action_reasoning}")


# ─── self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import gzip
    from pathlib import Path
    from intake_parser import intake

    RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
    RUN_ID = "pyca/bcrypt_.github/workflows/wheel-builder.yml_82_1"

    print(f"Finding {RUN_ID}...")
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run.get("_id") == RUN_ID:
                break
        else:
            raise SystemExit("Run not found.")

    event = intake(run)
    print(f"Run: {event.repo} #{event.run_number}")
    print(f"Branch: {event.branch}")
    print(f"Commit: {event.commit_sha} — {event.commit_title}")

    gt = scrape_ground_truth(event)
    print_ground_truth(gt)
