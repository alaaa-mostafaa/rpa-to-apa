import json
from pathlib import Path

results = json.loads(Path('data/benchmark_1000_eig_vs_rpa.json').read_text())
unknowns = [r for r in results if r.get('ground_truth', {}).get('action') == 'UNKNOWN' and r.get('rpa', {}).get('judge', {}).get('verdict') == 'NOT_SCORABLE']

print(f"Total UNKNOWN cases: {len(unknowns)}")
print("Sample of UNKNOWN reasoning from scraper:")
for u in unknowns[:15]:
    print(f"- {u['repo']} ({u['commit']}): {u['ground_truth'].get('reasoning')}")
