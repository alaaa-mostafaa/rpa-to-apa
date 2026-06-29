import pytest
from faf import GitHubAdapter, KubernetesAdapter

def test_github_adapter_parse_raw():
    adapter = GitHubAdapter()
    raw_run = {
        "_id": "gh-123",
        "repository_name": "org/repo",
        "metadata": {
            "head_branch": "main",
            "conclusion": "failure",
            "head_commit": {"message": "Test commit"}
        },
        "log_insights": [
            {
                "file": "test.yml",
                "steps": [{"error": "Test error"}]
            }
        ]
    }
    
    event = adapter.parse_raw(raw_run)
    assert event.source == "github"
    assert event.run_id == "gh-123"
    assert "error_text" in event.available_signals
    assert len(event.failed_steps) == 1
    assert event.failed_steps[0].error_text == "Test error"

def test_kubernetes_adapter_parse_raw():
    adapter = KubernetesAdapter()
    payload = {
        "namespace": "default",
        "pod_name": "app-pod",
        "phase": "Failed",
        "reason": "OOMKilled",
        "logs": "Memory exceeded",
        "events": ["Warning: OOM"],
        "labels": {"commit_sha": "abc"}
    }
    
    event = adapter.parse_raw(payload)
    assert event.source == "kubernetes"
    assert event.metadata["commit_sha"] == "abc"
    assert "OOMKilled" in event.failed_steps[0].error_text
