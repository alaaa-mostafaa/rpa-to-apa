# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
import json

data = json.load(open('data/l3_retrieval_eval.json'))
cases = data['cases']

hurt = [c for c in cases if c['l3_outcome'] == 'HURT']
rescue = [c for c in cases if c['l3_outcome'] == 'RESCUE']

print('HURT cases:', len(hurt))
for h in hurt[:15]:
    gt = h.get('gt_category', '')
    apa = h.get('apa_category', '')
    pt = h.get('prior_top', '')
    pr = h.get('prior_rank_gt', '')
    nn = h.get('n_neighbours', 0)
    print(f'  gt={gt:<28} apa={apa:<28} prior_top={pt:<28} rank={pr} nbrs={nn}')

print()
print('RESCUE cases:', len(rescue))
for r in rescue[:10]:
    gt = r.get('gt_category', '')
    apa = r.get('apa_category', '')
    pt = r.get('prior_top', '')
    pr = r.get('prior_rank_gt', '')
    print(f'  gt={gt:<28} apa={apa:<28} prior_top={pt:<28} rank={pr}')

print()
correct_cases = [c for c in cases if c['apa_verdict'] == 'CORRECT']
prior_agrees = [c for c in correct_cases if c['prior_top'] == c['gt_category']]
print(f'APA correct: {len(correct_cases)}, prior agrees: {len(prior_agrees)} ({len(prior_agrees)/max(1,len(correct_cases)):.1%})')

# The HURT condition is too aggressive — it fires when prior_top != gt even if prior_rank is low
# Check: of the 40 HURT cases, how many have prior_rank <= 2?
hurt_rank = [c for c in hurt if c.get('prior_rank_gt', 99) <= 2]
print(f'HURT cases where prior still ranks gt in top-2: {len(hurt_rank)} (not actually hurt)')

# What if we only count HURT when prior_rank >= 5 (prior actively buries gt)?
hurt_severe = [c for c in hurt if c.get('prior_rank_gt', 0) >= 5]
print(f'HURT cases where prior buries gt (rank>=5): {len(hurt_severe)}')

# Category breakdown of HURT
from collections import Counter
hurt_gt_cats = Counter(c.get('gt_category', 'unknown') for c in hurt)
print('\nHURT by gt_category:')
for cat, n in hurt_gt_cats.most_common():
    print(f'  {cat:<30} {n}')
