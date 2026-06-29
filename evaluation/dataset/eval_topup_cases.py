# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Run RPA + APA on the 47 mined top-up cases (reusing run_case from run_eval_500),
# using the GT already embedded in each mined case (no re-scrape). Emit eval_big100-format
# records and append to data/eval_big100.jsonl -> ~300 corpus.
import json, gzip, os, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
os.environ.setdefault("CI_AGENT_CLASSIFY_MODEL", "deepseek-reasoner")
import run_eval_500 as R
from src.apa.llm_config import make_client
from archive.ground_truth_scraper import GroundTruth

OUT = "data/eval_topup_records.jsonl"
existing = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        if l.strip(): existing.add(json.loads(l)["run_id"])
# also skip run_ids already in the main corpus
for l in open("data/eval_big100.jsonl", encoding="utf-8"):
    if l.strip(): existing.add(json.loads(l)["run_id"])

cases = [json.loads(l) for l in gzip.open("data/dataset_topup.jsonl.gz", "rt", encoding="utf-8") if l.strip()]
client = make_client()
print(f"running RPA+APA on {len(cases)} mined cases (skipping {len(existing)} already done)")

done = 0
for c in cases:
    rid = c["intake"]["run_id"]
    if rid in existing:
        continue
    gtd = c.get("ground_truth", {})
    gt = GroundTruth(run_id=rid, repo=c["intake"]["repo"], branch=c["intake"].get("branch", ""),
                     failing_sha=c["intake"].get("commit_sha", ""))
    gt.developer_action = gtd.get("action", "UNKNOWN")
    gt.developer_action_reasoning = gtd.get("reasoning", "")
    gt.classification_method = gtd.get("method", "none")
    try:
        rec = R.run_case(c, client, gt=gt)
    except Exception as e:
        print(f"  ! {rid[:50]}: {type(e).__name__}: {e}")
        continue
    rec["mined_topup"] = True
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done += 1
    rp = (rec.get("rpa", {}).get("prediction") or {}).get("category")
    ap = (rec.get("apa", {}).get("prediction") or {}).get("category")
    print(f"  [{done}] {rec['repo'][:30]:30} RPA={rp} APA={ap} GT={gt.developer_action}")

print(f"\ndone {done}. records in {OUT}")
