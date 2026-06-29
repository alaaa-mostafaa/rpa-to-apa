#!/usr/bin/env python3
# demo_launch.py
# ─────────────────────────────────────────────────────────────────
# ONE-COMMAND demo launcher.
#
# Usage:
#   python demo_launch.py --token ghp_YOUR_TOKEN --repo owner/repo-name
#
# What it does:
#   1. Starts the webhook handler (port 8090) with your GitHub token
#   2. Starts ngrok to expose port 8090 to the internet
#   3. Prints the public URL to paste into GitHub webhook settings
#   4. Opens the cicd-pipeline.html demo in your browser
#   5. Tails the live audit log so you see each decision arrive
# ─────────────────────────────────────────────────────────────────

import argparse
import json
import os
import subprocess
import sys
import time
import threading
import webbrowser
from pathlib import Path

# ── colours ──────────────────────────────────────────────────────
G  = "\033[92m"   # green
B  = "\033[94m"   # blue
Y  = "\033[93m"   # yellow
R  = "\033[91m"   # red
C  = "\033[96m"   # cyan
W  = "\033[97m"   # white bold
DIM= "\033[2m"
RST= "\033[0m"

def banner():
    print(f"""
{C}╔══════════════════════════════════════════════════════════╗
║      CI TRIAGE AGENT  —  Live Demo Launcher              ║
║      RPA → APA  |  GitHub + K8s Integration              ║
╚══════════════════════════════════════════════════════════╝{RST}
""")

def section(title):
    print(f"\n{B}{'─'*60}{RST}")
    print(f"{W}  {title}{RST}")
    print(f"{B}{'─'*60}{RST}")

def ok(msg):   print(f"  {G}✔{RST}  {msg}")
def info(msg): print(f"  {C}ℹ{RST}  {msg}")
def warn(msg): print(f"  {Y}⚠{RST}  {msg}")
def err(msg):  print(f"  {R}✘{RST}  {msg}")

# ── ngrok tunnel ─────────────────────────────────────────────────

def start_ngrok(port: int) -> str:
    """Start ngrok and return the public HTTPS URL."""
    info("Starting ngrok tunnel...")
    try:
        # Start ngrok in background
        proc = subprocess.Popen(
            ["ngrok", "http", str(port), "--log=stdout", "--log-format=json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # Give ngrok 3 seconds to establish the tunnel
        time.sleep(3)

        # Query the ngrok API for the tunnel URL
        import urllib.request
        try:
            with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=5) as resp:
                data = json.loads(resp.read())
                tunnels = data.get("tunnels", [])
                for t in tunnels:
                    url = t.get("public_url", "")
                    if url.startswith("https://"):
                        ok(f"ngrok tunnel: {G}{url}{RST}")
                        return url, proc
        except Exception as e:
            warn(f"Could not query ngrok API: {e}")

        # Fallback: parse from stdout
        warn("Trying to parse ngrok URL from output...")
        return None, proc

    except FileNotFoundError:
        err("ngrok not found in PATH")
        err("Install from: https://ngrok.com/download")
        return None, None


# ── webhook handler ───────────────────────────────────────────────

def start_webhook_handler(token: str, port: int, k8s_ns: str, k8s_dep: str) -> subprocess.Popen:
    """Start webhook_handler.py as a subprocess."""
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = token

    cmd = [
        sys.executable, "webhook_handler.py",
        "--port", str(port),
    ]
    if k8s_ns:
        cmd += ["--k8s-namespace", k8s_ns]
    if k8s_dep:
        cmd += ["--k8s-deployment", k8s_dep]

    proc = subprocess.Popen(cmd, env=env, text=True)
    time.sleep(1.5)
    ok(f"Webhook handler running on port {port}")
    return proc


# ── audit log tail ────────────────────────────────────────────────

def tail_audit_log(stop_event: threading.Event):
    """Continuously print new audit log entries as they arrive."""
    from datetime import datetime, timezone

    log_dir = Path("audit_logs")
    seen_ids = set()

    ACTION_COLOR = {
        "BLOCK_MERGE": R,
        "RETRY":       Y,
        "QUARANTINE":  Y,
        "INVESTIGATE": C,
        "IGNORE":      DIM,
        "PROMOTE":     G,
        "PAUSE":       Y,
        "ROLLBACK":    R,
    }
    TIER_COLOR = {"T0": DIM, "T1": Y, "T2": G}

    while not stop_event.is_set():
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = log_dir / f"audit_{today}.jsonl"

        if log_file.exists():
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rid = rec.get("id")
                        if rid in seen_ids:
                            continue
                        seen_ids.add(rid)

                        # Pretty-print the new decision
                        action = rec.get("action", "?")
                        tier   = rec.get("trust_tier", "?")
                        cat    = rec.get("category", "?")
                        conf   = rec.get("confidence", 0)
                        repo   = rec.get("repo", "?")
                        branch = rec.get("branch", "?")
                        dep    = rec.get("deployment_action", "—")
                        ts     = rec.get("timestamp", "")[:19].replace("T", " ")

                        ac = ACTION_COLOR.get(action, W)
                        tc = TIER_COLOR.get(tier, W)

                        print(f"\n{B}{'═'*60}{RST}")
                        print(f"  {G}▶ NEW DECISION RECEIVED{RST}  {DIM}{ts}{RST}")
                        print(f"  {W}Repo:{RST}     {repo}  {DIM}({branch}){RST}")
                        print(f"  {W}Category:{RST} {cat}  {DIM}conf={conf:.0%}{RST}")
                        print(f"  {W}Action:{RST}   {ac}{action}{RST}")
                        print(f"  {W}Trust:{RST}    {tc}{tier}{RST}  (Canary: {dep})")

                        # GitHub dispatch status
                        gh = (rec.get("dispatch") or {}).get("github", {})
                        if gh.get("dispatched"):
                            print(f"  {G}GitHub:{RST}   Check run posted ✔  PR comment posted ✔")
                        elif gh.get("check_run", {}).get("dry_run"):
                            print(f"  {Y}GitHub:{RST}   [DRY-RUN] Would post check run + PR comment")
                        else:
                            reason = gh.get("reason", "not dispatched")
                            print(f"  {DIM}GitHub:{RST}   {reason}")

                        # K8s dispatch status
                        k8s = (rec.get("dispatch") or {}).get("k8s", {})
                        if k8s.get("dispatched"):
                            result = k8s.get("result", {})
                            print(f"  {G}K8s:{RST}      {dep} → applied ✔  {DIM}{result}{RST}")
                        elif k8s.get("deployment_action"):
                            # Show what WOULD be sent
                            dep_action = k8s.get("deployment_action", "PAUSE")
                            patch = _k8s_dry_run_patch(dep_action)
                            print(f"  {C}K8s:{RST}      Would PATCH deployment: {DIM}{json.dumps(patch)}{RST}")
                        else:
                            print(f"  {DIM}K8s:{RST}      not configured")

                        print(f"  {DIM}Audit ID: {rid[:16]}...{RST}")
                        print(f"{B}{'═'*60}{RST}")

            except Exception:
                pass

        time.sleep(1)


def _k8s_dry_run_patch(action: str) -> dict:
    """Return the K8s PATCH body that would be applied for a given action."""
    if action == "PROMOTE":
        return {"spec": {"paused": False}}
    elif action == "ROLLBACK":
        return {"metadata": {"annotations": {"deployment.kubernetes.io/revision": "prev"}}}
    else:
        return {"spec": {"paused": True}}


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CI Triage Agent demo launcher")
    parser.add_argument("--token",    required=True, help="GitHub PAT token (ghp_...)")
    parser.add_argument("--repo",     required=True, help="GitHub repo (owner/name)")
    parser.add_argument("--port",     type=int, default=8090)
    parser.add_argument("--k8s-namespace",  default="production")
    parser.add_argument("--k8s-deployment", default="my-app")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    banner()
    processes = []

    # ── 1. Start webhook handler ──────────────────────────────────
    section("STEP 1: Starting Webhook Handler")
    wh_proc = start_webhook_handler(
        args.token, args.port,
        args.k8s_namespace, args.k8s_deployment,
    )
    processes.append(wh_proc)

    # ── 2. Start ngrok ────────────────────────────────────────────
    section("STEP 2: Opening ngrok Tunnel")
    ngrok_url, ngrok_proc = start_ngrok(args.port)
    if ngrok_proc:
        processes.append(ngrok_proc)

    # ── 3. Print setup instructions ───────────────────────────────
    section("STEP 3: Configure GitHub Webhook")
    if ngrok_url:
        webhook_url = f"{ngrok_url}/webhook"
        print(f"""
  {W}Go to:{RST}  github.com/{args.repo}/settings/hooks/new

  {Y}Payload URL:{RST}
    {G}{webhook_url}{RST}

  {Y}Content type:{RST}  application/json
  {Y}Events:{RST}        Workflow runs  (or 'send me everything')
  {Y}Active:{RST}        ✔
""")
    else:
        warn("Could not get ngrok URL. Check http://localhost:4040 manually.")

    # ── 4. Open demo UI ───────────────────────────────────────────
    section("STEP 4: Opening Demo UI")
    html_path = Path("cicd-pipeline.html").resolve()
    if html_path.exists() and not args.no_browser:
        webbrowser.open(f"file:///{html_path}")
        ok(f"Opened {html_path.name} in browser")
    else:
        info(f"Open manually: {html_path}")

    # ── 5. Print demo script ──────────────────────────────────────
    section("DEMO FLOW")
    print(f"""
  {W}1.{RST} Show the pipeline visualizer in the browser (already open)
  {W}2.{RST} Walk through the bcrypt case: Intake → Extraction → Classification

  {W}3.{RST} Switch to this terminal  ← live feed will appear here

  {W}4.{RST} Open github.com/{args.repo} in browser
     Create a new branch, push the broken auth.py, open a PR

  {W}5.{RST} Watch the CI action fail (takes ~30s)

  {W}6.{RST} Watch this terminal: the agent classifies the failure
     Trust Tier, Action, GitHub + K8s dispatch all shown here

  {W}7.{RST} Flip back to the PR on GitHub
     {G}The agent has posted a comment automatically!{RST}

  {Y}TIP:{RST} Have github.com/{args.repo}/pulls open on a second monitor
       so supervisors see the comment appear in real time.
""")

    section("LIVE AUDIT FEED  (waiting for webhooks...)")
    info(f"Webhook endpoint: {ngrok_url or 'http://localhost:' + str(args.port)}/webhook")
    info("Push a commit to your test repo to trigger the demo")
    print()

    # ── 6. Tail audit log ─────────────────────────────────────────
    stop = threading.Event()
    tail_thread = threading.Thread(target=tail_audit_log, args=(stop,), daemon=True)
    tail_thread.start()

    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print(f"\n\n{Y}Shutting down...{RST}")
        stop.set()
        for proc in processes:
            proc.terminate()
        print(f"{G}Done. Audit logs saved in audit_logs/{RST}")


if __name__ == "__main__":
    main()
