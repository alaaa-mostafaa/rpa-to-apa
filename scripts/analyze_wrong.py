import json

data = json.load(open('data/benchmark_1000_eig_vs_rpa.json'))
wrong = [r for r in data if r.get('apa_eig',{}).get('judge',{}).get('verdict') == 'WRONG']
confidences = [r.get('apa_eig',{}).get('prediction',{}).get('confidence',0) for r in wrong]
print(f'Wrong cases: {len(wrong)}, avg confidence: {sum(confidences)/max(len(confidences),1):.2f}')

config_wrong = [r for r in wrong if r.get('apa_eig',{}).get('prediction',{}).get('category') == 'CONFIG_ERROR']
print(f'\nCONFIG_ERROR wrong ({len(config_wrong)}):')
for r in config_wrong[:3]:
    gt_action = r['ground_truth']['action']
    apa_conf = r['apa_eig']['prediction'].get('confidence', 0)
    judge_reason = r['apa_eig']['judge']['reasoning'][:250]
    print(f'  GT={gt_action} | conf={apa_conf} | judge={judge_reason}')
    print()

env_wrong = [r for r in wrong if r.get('apa_eig',{}).get('prediction',{}).get('category') == 'ENV_FLAKINESS']
print(f'ENV_FLAKINESS wrong ({len(env_wrong)}):')
for r in env_wrong[:2]:
    gt_action = r['ground_truth']['action']
    apa_conf = r['apa_eig']['prediction'].get('confidence', 0)
    judge_reason = r['apa_eig']['judge']['reasoning'][:250]
    print(f'  GT={gt_action} | conf={apa_conf} | judge={judge_reason}')
    print()
