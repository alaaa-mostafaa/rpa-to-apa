"""Quick smoke test: run remediation on one real benchmark case."""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
import sys

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
load_dotenv()

os.environ["LLM_PROVIDER"] = "deepseek"

from src.apa.remediation import generate_remediation
from src.apa.llm_config import make_client

# Load one CORRECT case from the benchmark to see what remediation looks like
data = json.load(open("data/benchmark_1000_eig_vs_rpa.json"))

# Find a case where APA got it right
for r in data:
    if r["apa_eig"]["judge"]["verdict"] == "CORRECT":
        case = r
        break

print(f"Testing on: {case['repo']}")
print(f"  APA category: {case['apa_eig']['prediction']['category']}")
print(f"  APA reasoning: {case['apa_eig']['prediction'].get('reasoning', '')[:150]}")
print(f"  GT action: {case['ground_truth']['action']}")
print()

# Build a minimal agent_state from what we have in the benchmark result
classification = case["apa_eig"]["prediction"]
agent_state = {
    "error_lines": [],
    "commit_diff": {},
    "changed_files": [],
    "dependency_changes": {},
    "workflow_contents": [],
    "semantic_diff_links": [],
}

client = make_client()
plan = generate_remediation(classification, agent_state, client)

print("=== REMEDIATION PLAN ===")
print(plan.to_markdown())
print()
print("=== JSON ===")
print(plan.to_json())
