import json
import time
import os
from pathlib import Path
from agent import run_agent
from decision_layer import enrich_result
from webhook_handler import append_audit_log

def main():
    # Keys are read from the environment (.env). Never hardcode credentials.
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("Set DEEPSEEK_API_KEY (e.g. in a .env file) before running the demo.")
    # GITHUB_TOKEN is optional (enables live commit-diff / run-history fetching).
    if not os.environ.get("CI_AGENT_MODEL"):
        os.environ["CI_AGENT_MODEL"] = "deepseek-chat"
        
    # We will pick a specific case from the dataset that is known to be a code regression
    # Or just let the user provide the path. Let's provide a good default.
    dataset_dir = Path("streaming_cases_10")
    
    # Try to find case 4, which might be a code regression or something interesting
    target_case_dir = dataset_dir / "case_04_astro_deadnix_astro_deadnix_.github_workflows_ci.yml_3_1"
    run_json = target_case_dir / "run_full.json"
    
    # If case 4 doesn't exist, just pick the first one that has a run_full.json
    if not run_json.exists():
        run_json = next(dataset_dir.rglob("run_full.json"), None)
        
    if not run_json:
        print("Could not find any run_full.json in streaming_cases_10")
        return
        
    print(f"Loading dataset case from: {run_json}")
    with open(run_json, "r", encoding="utf-8") as f:
        raw_run = json.load(f)
        
    print(f"Running agent on: {raw_run.get('repository_name')} run #{raw_run.get('run_number')}...")
    t0 = time.monotonic()
    
    # Run the full agent pipeline
    agent_result = run_agent(raw_run)
    duration = time.monotonic() - t0
    
    cat = agent_result.get("classification", {}).get("category", "?")
    conf = agent_result.get("classification", {}).get("confidence", 0) * 100
    print(f"\nAgent completed in {duration:.1f}s: category={cat} confidence={conf:.0f}%")
    
    # Build context for the decision layer
    head_sha = raw_run.get("metadata", {}).get("head_sha", "")
    branch = raw_run.get("metadata", {}).get("head_branch", "")
    repo_full = raw_run.get("repository_name", "")
    workflow_name = raw_run.get("metadata", {}).get("name", "CI")
    run_number = raw_run.get("run_number", 0)
    
    context = {
        "repo": repo_full,
        "run_id": f"{repo_full}_{workflow_name}_{run_number}",
        "commit_sha": head_sha,
        "pr_number": None,
        "protected_branch": branch in ("main", "master", "develop"),
        "agent_mode": "APA (Dataset Demo)",
    }
    
    # Enrich and log to audit!
    enriched = enrich_result(agent_result, context=context)
    decision = enriched.get("decision", {})
    audit = enriched.get("audit", {})
    
    audit["commit_sha"] = head_sha
    audit["branch"] = branch
    audit["workflow_name"] = workflow_name
    audit["agent_duration_sec"] = round(duration, 2)
    audit["dispatch"] = {"github": {"dispatched": False, "reason": "dataset demo"}}
    
    log_file = append_audit_log(audit)
    print(f"\nAudit logged to {log_file}")
    print("\nCheck your demo UI! The new case should be animating across the screen right now.")

if __name__ == "__main__":
    main()
