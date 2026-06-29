# webhook_handler.py
# ─────────────────────────────────────────────────────────────────────
# Webhook server that receives GitHub Actions events, runs the CI
# triage agent, and dispatches decisions to GitHub and/or Kubernetes.
#
# Architecture:
#   GitHub webhook (workflow_run.completed)
#       |
#       v
#   webhook_handler.py (this file)
#       |
#       +-- agent.run_agent(raw_run)          # classify failure
#       +-- decision_layer.enrich_result()    # add triage/trust/policy
#       +-- GitHubDispatcher.dispatch()       # check run + PR comment
#       +-- K8sDispatcher.dispatch()          # canary rollout control
#       +-- append to audit log file
#
# Usage:
#   # Development (dry-run, no real API calls):
#   python webhook_handler.py --dry-run --port 8090
#
#   # Production:
#   GITHUB_TOKEN=ghp_... python webhook_handler.py --port 8090
#
#   # With K8s canary control:
#   GITHUB_TOKEN=ghp_... python webhook_handler.py \
#       --k8s-namespace production --k8s-deployment my-app
#
# GitHub webhook setup:
#   1. Go to your repo Settings > Webhooks > Add webhook
#   2. Payload URL: https://your-server:8090/webhook
#   3. Content type: application/json
#   4. Secret: (set WEBHOOK_SECRET env var to match)
#   5. Events: select "Workflow runs"
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ─── config ──────────────────────────────────────────────────────────

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
AUDIT_LOG_DIR = os.environ.get("AUDIT_LOG_DIR", "audit_logs")
PORT = int(os.environ.get("WEBHOOK_PORT", "8090"))

# K8s config (optional)
K8S_NAMESPACE = os.environ.get("K8S_NAMESPACE", "")
K8S_DEPLOYMENT = os.environ.get("K8S_DEPLOYMENT", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("webhook_handler")

# ─── import path bootstrap ──────────────────────────────────────────
# Allows this file to work when executed from project root as:
#   python -m demos.webhook_handler --port 8090
# and also supports older local imports inside src/apa such as:
#   import agent
#   from intake_parser import ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
APA_DIR = SRC_DIR / "apa"

for path in (PROJECT_ROOT, SRC_DIR, APA_DIR):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)



# ─── signature verification ──────────────────────────────────────────

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature."""
    if not secret:
        return True  # No secret configured = skip verification
    if not signature:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ─── audit log persistence ──────────────────────────────────────────

def append_audit_log(audit: Dict[str, Any], directory: str = AUDIT_LOG_DIR) -> Path:
    """Append an audit record to the daily JSONL log file."""
    log_dir = Path(directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"audit_{today}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return log_file


# ─── core pipeline ───────────────────────────────────────────────────

def handle_workflow_run(
    payload: Dict[str, Any],
    *,
    dry_run: bool = False,
    github_token: str = "",
    k8s_namespace: str = "",
    k8s_deployment: str = "",
) -> Dict[str, Any]:
    """
    Process a workflow_run webhook event end-to-end.

    1. Extract run metadata from the webhook payload
    2. Run the CI triage agent (classification)
    3. Enrich with decision layer (triage + trust + policy)
    4. Dispatch to GitHub (check run + PR comment)
    5. Dispatch to K8s (canary control, if configured)
    6. Persist audit record

    Returns the full pipeline result dict.
    """
    from src.apa.decision_layer import (
        enrich_result,
        GitHubDispatcher,
        K8sDispatcher,
        Dispatcher,
    )

    workflow_run = payload.get("workflow_run", {})
    action = payload.get("action", "")
    repo_full = workflow_run.get("repository", {}).get("full_name", "")
    if not repo_full:
        repo_full = payload.get("repository", {}).get("full_name", "")

    run_id = workflow_run.get("id", "")
    run_number = workflow_run.get("run_number", 0)
    head_sha = workflow_run.get("head_sha", "")
    branch = workflow_run.get("head_branch", "")
    conclusion = workflow_run.get("conclusion", "")
    workflow_name = workflow_run.get("name", "")
    event_type = workflow_run.get("event", "")

    # Extract PR number if available
    pr_number = None
    prs = workflow_run.get("pull_requests", [])
    if prs:
        pr_number = prs[0].get("number")

    log.info(
        "Processing workflow_run: repo=%s run=#%s branch=%s conclusion=%s",
        repo_full, run_number, branch, conclusion,
    )

    # Only process completed failures
    if action != "completed":
        return {"skipped": True, "reason": f"action={action}, not completed"}
    if conclusion != "failure":
        return {"skipped": True, "reason": f"conclusion={conclusion}, not failure"}

    # ── Step 1: Build a raw_run dict from the webhook payload ──────
    # The agent expects a raw_run dict from the GHALogs dataset format.
    # We adapt the webhook payload to match that schema.
    raw_run = _webhook_to_raw_run(payload)

    # ── Step 2: Run the CI triage agent ───────────────────────────
    t0 = time.monotonic()
    try:
        from src.apa.agent import run_agent
        agent_result = run_agent(raw_run)
    except Exception as e:
        log.error("Agent failed: %s", e)
        agent_result = {
            "classification": {
                "category": "UNKNOWN",
                "severity": "MODERATE",
                "confidence": 0.0,
                "reasoning": f"Agent error: {e}",
            },
            "beliefs": {},
            "tools_used": [],
            "steps_taken": 0,
            "fast_path": False,
            "preprocessing_summary": {},
        }
    agent_duration = time.monotonic() - t0
    log.info(
        "Agent completed in %.1fs: category=%s confidence=%.0f%%",
        agent_duration,
        agent_result.get("classification", {}).get("category", "?"),
        agent_result.get("classification", {}).get("confidence", 0) * 100,
    )

    # ── Step 3: Enrich with decision layer ────────────────────────
    context = {
        "repo": repo_full,
        "run_id": f"{repo_full}_{workflow_name}_{run_number}",
        "commit_sha": head_sha,
        "pr_number": pr_number,
        "protected_branch": branch in ("main", "master", "develop"),
        "agent_mode": "APA" if agent_result.get("tools_used") else "RPA",
    }
    enriched = enrich_result(agent_result, context=context)
    decision = enriched.get("decision", {})
    audit = enriched.get("audit", {})

    # Add webhook-specific fields to audit
    audit["commit_sha"] = head_sha
    audit["pr_number"] = pr_number
    audit["branch"] = branch
    audit["workflow_name"] = workflow_name
    audit["agent_duration_sec"] = round(agent_duration, 2)
    decision["commit_sha"] = head_sha
    decision["pr_number"] = pr_number

    log.info(
        "Decision: action=%s trust=%s canary=%s",
        decision.get("action"), decision.get("trust_tier"),
        decision.get("deployment_action"),
    )

    # ── Step 4: Dispatch to GitHub ────────────────────────────────
    github_dispatcher = GitHubDispatcher(
        token=github_token or GITHUB_TOKEN,
        repo=repo_full,
        dry_run=dry_run,
    )
    github_receipt = github_dispatcher.dispatch(decision, audit)
    log.info("GitHub dispatch: %s", "OK" if github_receipt.get("dispatched") else "skipped")

    # ── Step 5: Dispatch to K8s (if configured) ───────────────────
    k8s_receipt = {"dispatched": False, "reason": "not configured"}
    if k8s_namespace and k8s_deployment:
        k8s_dispatcher = K8sDispatcher(
            namespace=k8s_namespace,
            deployment=k8s_deployment,
            dry_run=dry_run,
        )
        k8s_receipt = k8s_dispatcher.dispatch(decision, audit)
        log.info("K8s dispatch: %s", "OK" if k8s_receipt.get("dispatched") else "skipped")

    # ── Step 6: Persist audit record ──────────────────────────────
    audit["dispatch"] = {
        "github": github_receipt,
        "k8s": k8s_receipt,
    }
    log_file = append_audit_log(audit)
    log.info("Audit logged to %s", log_file)

    return {
        "processed": True,
        "repo": repo_full,
        "run_number": run_number,
        "head_sha": head_sha[:8],
        "classification": agent_result.get("classification", {}),
        "decision": {
            "action": decision.get("action"),
            "trust_tier": decision.get("trust_tier"),
            "deployment_action": decision.get("deployment_action"),
        },
        "dispatch": {
            "github": {"dispatched": github_receipt.get("dispatched", False)},
            "k8s": {"dispatched": k8s_receipt.get("dispatched", False)},
        },
        "agent_duration_sec": round(agent_duration, 2),
    }


def _webhook_to_raw_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a GitHub webhook workflow_run payload into the raw_run
    dict format expected by agent.run_agent().

    The agent was built for the GHALogs dataset. This adapter maps
    the webhook payload fields to that schema so the agent works
    unchanged.
    """
    wr = payload.get("workflow_run", {})
    repo = payload.get("repository", {})

    # Extract failed jobs from the jobs list (if provided by a
    # subsequent /actions/runs/{id}/jobs API call, or pre-fetched)
    jobs = wr.get("jobs", [])
    failed_jobs = [j for j in jobs if j.get("conclusion") == "failure"]

    return {
        "_id": f"{repo.get('full_name', '')}_{wr.get('name', '')}_{wr.get('run_number', 0)}_{wr.get('run_attempt', 1)}",
        "repository_name": repo.get("full_name", ""),
        "workflow_path": wr.get("path", ""),
        "run_number": wr.get("run_number", 0),
        "run_attempt": wr.get("run_attempt", 1),
        "metadata": {
            "event": wr.get("event", ""),
            "status": wr.get("status", ""),
            "conclusion": wr.get("conclusion", ""),
            "head_branch": wr.get("head_branch", ""),
            "head_commit": {
                "id": wr.get("head_sha", ""),
                "message": (wr.get("head_commit") or {}).get("message", ""),
                "author": (wr.get("head_commit") or {}).get("author", {}),
            },
            "actor": wr.get("actor", {}),
            "run_started_at": wr.get("run_started_at", ""),
            "updated_at": wr.get("updated_at", ""),
        },
        "log_insights": [] # The agent will fetch jobs/logs via GitHub API if not provided here
    }


# ─── HTTP server (stdlib, no Flask dependency) ───────────────────────

def create_app(
    dry_run: bool = False,
    k8s_namespace: str = "",
    k8s_deployment: str = "",
):
    """Create an HTTP request handler class with the given config."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class WebhookHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            """Health check endpoint."""
            if self.path == "/health":
                self._respond(200, {"status": "ok", "dry_run": dry_run})
            else:
                self._respond(200, {
                    "service": "CI Triage Agent Webhook Handler",
                    "endpoints": {
                        "/webhook": "POST - GitHub webhook receiver",
                        "/health": "GET - health check",
                    },
                    "dry_run": dry_run,
                })

        def do_POST(self):
            """Handle incoming webhook."""
            if self.path != "/webhook":
                self._respond(404, {"error": "not found"})
                return

            # Read body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Verify signature
            signature = self.headers.get("X-Hub-Signature-256", "")
            if not verify_signature(body, signature, WEBHOOK_SECRET):
                log.warning("Invalid webhook signature")
                self._respond(401, {"error": "invalid signature"})
                return

            # Parse payload
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._respond(400, {"error": "invalid JSON"})
                return

            # Check event type
            event_type = self.headers.get("X-GitHub-Event", "")
            if event_type != "workflow_run":
                self._respond(200, {"skipped": True, "reason": f"event={event_type}"})
                return

            # Process
            try:
                result = handle_workflow_run(
                    payload,
                    dry_run=dry_run,
                    k8s_namespace=k8s_namespace,
                    k8s_deployment=k8s_deployment,
                )
                status = 200 if result.get("processed") or result.get("skipped") else 500
                self._respond(status, result)
            except Exception as e:
                log.exception("Webhook processing failed")
                self._respond(500, {"error": str(e)})

        def _respond(self, status: int, body: dict):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body, indent=2).encode("utf-8"))

        def log_message(self, format, *args):
            """Route HTTP logs through our logger."""
            log.debug(format, *args)

    return HTTPServer, WebhookHandler


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CI Triage Agent webhook handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Dry-run mode (no real API calls):
  python webhook_handler.py --dry-run --port 8090

  # Production with GitHub integration:
  GITHUB_TOKEN=ghp_... python webhook_handler.py --port 8090

  # With Kubernetes canary control:
  GITHUB_TOKEN=ghp_... python webhook_handler.py \\
      --k8s-namespace production --k8s-deployment my-app

  # Test with a simulated webhook payload:
  python webhook_handler.py --simulate test_payload.json
""",
    )
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't make real API calls to GitHub/K8s")
    parser.add_argument("--k8s-namespace", default=K8S_NAMESPACE)
    parser.add_argument("--k8s-deployment", default=K8S_DEPLOYMENT)
    parser.add_argument("--simulate", type=Path, default=None,
                        help="Process a saved webhook payload JSON and exit")
    args = parser.parse_args()

    if args.simulate:
        # Offline simulation mode: process a saved payload
        if not args.simulate.exists():
            print(f"ERROR: {args.simulate} not found", file=sys.stderr)
            sys.exit(1)
        payload = json.loads(args.simulate.read_text(encoding="utf-8"))
        result = handle_workflow_run(
            payload,
            dry_run=True,  # Always dry-run in simulation
            k8s_namespace=args.k8s_namespace,
            k8s_deployment=args.k8s_deployment,
        )
        print(json.dumps(result, indent=2))
        return

    # Start HTTP server
    HTTPServer, Handler = create_app(
        dry_run=args.dry_run,
        k8s_namespace=args.k8s_namespace,
        k8s_deployment=args.k8s_deployment,
    )
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    mode = "DRY-RUN" if args.dry_run else "LIVE"
    log.info("CI Triage Agent webhook handler starting [%s] on port %d", mode, args.port)
    if args.k8s_namespace and args.k8s_deployment:
        log.info("K8s canary control: %s/%s", args.k8s_namespace, args.k8s_deployment)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
