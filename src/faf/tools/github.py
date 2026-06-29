from typing import Dict, Any

def inspect_commit_diff(state: dict) -> dict:
    event = state.get("event")
    sha = event.metadata.get("commit_sha", "unknown") if event else "unknown"
    return {
        "investigation_log": [f"[inspect_commit_diff] Mocked: Found 2 files changed for {sha} (source code)."]
    }

def check_run_history(state: dict) -> dict:
    return {
        "investigation_log": ["[check_run_history] Mocked: Previous runs on this branch were also failing."]
    }

def inspect_workflow_file(state: dict) -> dict:
    return {
        "investigation_log": ["[inspect_workflow_file] Mocked: No syntax errors found in YAML."]
    }

def inspect_pr_context(state: dict) -> dict:
    return {
        "investigation_log": ["[inspect_pr_context] Mocked: Not a PR run."]
    }

def inspect_dependency_changes(state: dict) -> dict:
    return {
        "investigation_log": ["[inspect_dependency_changes] Mocked: No dependency manifests changed."]
    }
