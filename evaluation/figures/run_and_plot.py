# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
import json
import gzip
import matplotlib.pyplot as plt
import asyncio
from dotenv import load_dotenv
import copy
import os

load_dotenv()
os.environ["LLM_PROVIDER"] = "deepseek"
os.environ["LLM_BASE_URL"] = "https://api.deepseek.com"

from src.apa.agent import build_agent_graph
from src.apa.bayesian_tracker_dual import DualTracker
from src.apa.intake_parser import RunEvent
from src.faf.llm import make_client

async def run_case_and_plot():
    candidate_repos = ["uniswap/governance-seatbelt", "arm-doe/pyart", "rroller/dahua", "pytorch/text"]
    
    # Load all cases at once so we can search
    print("Loading cases from GZ...")
    cases = {}
    with gzip.open("data/intake_logs_5000.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            repo = d.get("intake", {}).get("repo", "") or d.get("repository_name", "")
            if repo in candidate_repos:
                cases[repo] = d
                if len(cases) == len(candidate_repos):
                    break
                    
    client = make_client()
    graph = build_agent_graph()
    
    for repo in candidate_repos:
        if repo not in cases:
            continue
            
        print(f"\n--- Testing repo {repo} ---")
        raw_case = cases[repo]
        event_dict = raw_case.get("intake", {})
        defaults = {
            "source": "github",
            "run_number": 0, "attempt": 1, "actor": "", 
            "commit_message": event_dict.get("commit_title", ""), 
            "commit_author": "", "started_at": "", "duration_sec": 0
        }
        kwargs = {**defaults, **event_dict}
        event = RunEvent(**{k: v for k, v in kwargs.items() if k in RunEvent.__dataclass_fields__})
        
        # Preprocessing (RPA)
        tracker = DualTracker(mode="rpa", client=None)
        tracker.observe_branch(event.is_protected_branch, event.branch)
        tracker.observe_jobs(event.failed_jobs_count, event.n_jobs)
        tracker.observe_commit(event.commit_message or event.commit_title)
        tracker.observe_detection(event.failure_detection)
        
        initial_state = {
            "run_event": kwargs,
            "raw_run": raw_case,
            "api_key": client.api_key,
            "model": "deepseek-reasoner",
            "beliefs": dict(tracker.state.probabilities),
            "belief_history": list(tracker.state.history),
            "confidence": tracker.state.confidence(),
            "entropy": tracker.state.entropy(),
            "tools_available": [
                "deep_log_analysis",
                "inspect_failed_step_context",
                "inspect_commit_diff",
                "inspect_dependency_changes",
                "inspect_runner_environment",
                "check_run_history",
                "inspect_workflow_file",
                "search_similar_failures",
            ],
            "tools_called": [],
            "investigation_log": [],
            "current_step": 0,
            "done": False,
            "error_lines": getattr(event, 'error_text', "").splitlines()[:50] if getattr(event, 'error_text', "") else [],
            "classification": {}
        }
        
        print("Running graph...")
        final_state = await graph.ainvoke(initial_state)
        history = final_state.get("belief_history", [])
        tools = final_state.get("tools_called", [])
        
        print(f"Finished in {len(tools)} steps.")
        
        if len(tools) >= 2: # 2 tools means 3 total points (init + 2 steps)
            print(f"Selecting {repo} for plotting!")
            
            with open("scratch_case.json", "w") as f:
                # Need to safely dump belief state
                safe_history = []
                for h in history:
                    probs = getattr(h, "probabilities", {}) if hasattr(h, "probabilities") else h.get("probabilities", {})
                    safe_history.append({"probabilities": probs})
                json.dump(safe_history, f, indent=2)
                
            all_cats = set()
            parsed_history = []
            for h in history:
                probs = getattr(h, "probabilities", {}) if hasattr(h, "probabilities") else h.get("probabilities", {})
                parsed_history.append(probs)
                for k in probs:
                    all_cats.add(k)
                    
            final_beliefs = parsed_history[-1]
            top_cats = sorted(all_cats, key=lambda k: final_beliefs.get(k, 0), reverse=True)[:5]
            
            plt.figure(figsize=(10, 6))
            steps = list(range(len(parsed_history)))
            
            for cat in top_cats:
                probs = [ph.get(cat, 0) for ph in parsed_history]
                linewidth = 3.5 if cat == top_cats[0] else 1.5
                linestyle = '-' if cat == top_cats[0] else '--'
                plt.plot(steps, probs, label=cat, marker='o', linewidth=linewidth, linestyle=linestyle)
                
            plt.title(f"Bayesian Belief Evolution for CI Diagnosis ({repo})", fontsize=14, fontweight='bold')
            plt.xlabel("Investigation Steps", fontsize=12)
            plt.ylabel("Probability", fontsize=12)
            plt.ylim(0, 1.05)
            
            x_labels = ["Init (RPA Prior)"]
            for i in range(1, len(parsed_history)):
                if i-1 < len(tools):
                    tool_name = tools[i-1].replace("_", "\n")
                    x_labels.append(f"Step {i}\n({tool_name})")
                else:
                    x_labels.append(f"Step {i}")
                    
            plt.xticks(steps, x_labels, rotation=45, ha='right', fontsize=9)
            plt.legend(title="Failure Category", loc="center left", bbox_to_anchor=(1, 0.5))
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig("images/belief_evolution.png", dpi=300, bbox_inches='tight')
            print("Saved plot to images/belief_evolution.png")
            return

if __name__ == "__main__":
    asyncio.run(run_case_and_plot())
