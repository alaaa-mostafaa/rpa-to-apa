#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/build_large_corpus.py
#
# Builds a large eval corpus (like honest_eval_results.json) by finding all cases
# in benchmark_1000_eig_vs_rpa.json where the APA agent was judged CORRECT, and
# pulling their raw intake/extraction data from intake_logs_5000.jsonl.gz.
# The APA agent's prediction is used as the ground truth category for retrieval eval.

import json
import gzip
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    benchmark_path = base_dir / "data" / "benchmark_1000_eig_vs_rpa.json"
    logs_path = base_dir / "data" / "intake_logs_5000.jsonl.gz"
    output_path = base_dir / "data" / "large_eval_results.json"

    print(f"Reading benchmark results from {benchmark_path.name}...")
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    # Filter for cases where APA was CORRECT
    correct_cases = [r for r in benchmark_data if r.get("apa_eig", {}).get("judge", {}).get("verdict") == "CORRECT"]
    
    # Map run_id -> category
    category_map = {}
    for case in correct_cases:
        run_id = case.get("run_id")
        cat = case.get("apa_eig", {}).get("prediction", {}).get("category")
        if run_id and cat:
            category_map[run_id] = cat

    print(f"Found {len(category_map)} CORRECT cases.")

    print(f"Extracting raw data from {logs_path.name}...")
    corpus = []
    with gzip.open(logs_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            run_id = raw.get("intake", {}).get("run_id")
            if run_id in category_map:
                # Build the case format expected by bootstrap_chroma.py
                case = {
                    "case_label": f"{raw.get('intake', {}).get('repo')} — {raw.get('intake', {}).get('commit_title')}",
                    "intake": raw.get("intake", {}),
                    "extraction": raw.get("extraction", {}),
                    "classification": {
                        "category": category_map[run_id]
                    }
                }
                corpus.append(case)
                # Remove from map so we don't duplicate
                del category_map[run_id]

    print(f"Successfully extracted {len(corpus)} cases.")
    if category_map:
        print(f"Warning: {len(category_map)} cases were not found in the logs.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2)

    print(f"Wrote corpus to {output_path.name}")

if __name__ == "__main__":
    main()
