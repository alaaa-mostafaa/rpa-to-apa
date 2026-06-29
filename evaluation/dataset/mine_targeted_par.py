# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Parallel targeted miner: same gates/labels as mine_targeted.py, but scrapes
# candidates CONCURRENTLY (thread pool) since each scrape is network-bound. This is
# the big speedup — sequential scrapes wasted most time waiting on GitHub.
#
# Phase 1 (fast, single-thread): scan runs.json, apply context gate + metadata + dedup
#         + cheap preflight, collect a CANDIDATE list (no expensive scrape yet).
# Phase 2 (parallel): scrape_ground_truth on candidates with N workers; bucket via the
#         eval's substantive_fix_buckets; keep clean-GT (pr_files/llm/rubric); extract
#         log + write case until per-category targets are met.
import argparse, gzip, json, os, sys, io, contextlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import build_dataset as B
from src.apa.intake_parser import intake
from src.apa.log_extractor import extract_log_excerpt
from archive.ground_truth_scraper import quick_gt_preflight, scrape_ground_truth
from evals.coarse_eval import substantive_fix_buckets

ACTION_BUCKET = {"PIN_VERSION":"DEPENDENCY","DEPENDENCY_CHANGE":"DEPENDENCY",
                 "WORKFLOW_FIX":"CONFIG","CODE_FIX":"CODE","CODE_CHANGE":"CODE"}
CLEAN_METHODS = ("pr_files","llm","rubric")

def load_existing(paths):
    runs,repos=set(),set()
    for p in paths:
        if not os.path.exists(p): continue
        op=gzip.open if p.endswith(".gz") else open
        for l in op(p,"rt",encoding="utf-8"):
            if l.strip():
                r=json.loads(l); rid=(r.get("intake") or r).get("run_id") or r.get("run_id")
                repo=(r.get("intake") or r).get("repo") or r.get("repo")
                if rid: runs.add(str(rid))
                if repo: repos.add(repo)
    return runs,repos

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-path",default="data/runs_zenodo.json.gz")
    ap.add_argument("--out",default="data/dataset_topup.jsonl.gz")
    ap.add_argument("--need-dependency",type=int,default=30)
    ap.add_argument("--need-config",type=int,default=17)
    ap.add_argument("--scan-end",type=int,default=600_000)
    ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--candidate-batch",type=int,default=60,
                    help="how many candidates to scrape per parallel wave")
    args=ap.parse_args()
    assert os.environ.get("ZIP_URL"),"ZIP_URL not set"
    zip_path=os.environ["ZIP_URL"]
    win=os.environ.get("GT_FIX_WINDOW_DAYS","7")

    existing_runs,existing_repos=load_existing([
        "data/eval_big100.jsonl","data/dataset_remote_250.jsonl.gz",
        "data/dataset_remote_120.jsonl.gz","data/dataset_remote_next.jsonl.gz"])
    print(f"existing: {len(existing_runs)} runs / {len(existing_repos)} repos to avoid",flush=True)

    need={"DEPENDENCY":args.need_dependency,"CONFIG":args.need_config}
    got={"DEPENDENCY":0,"CONFIG":0}
    writer=B._CaseWriter(Path(args.out),"jsonl.gz")
    wlock=threading.Lock()
    tried_repos=set()
    print(f"target +{need['DEPENDENCY']} DEP +{need['CONFIG']} CONFIG | window={win}d | workers={args.workers}",flush=True)

    def scrape_and_build(run):
        """Full scrape + bucket + (if target) extract log + build case. Returns case or None."""
        meta=run.get("metadata") or {}
        repo_full=run.get("repository_name","")
        try:
            event=intake(run)
            with contextlib.redirect_stdout(io.StringIO()):
                gt=scrape_ground_truth(event)
        except Exception:
            return None
        method=getattr(gt,"classification_method","none")
        if method not in CLEAN_METHODS: return None
        action=getattr(gt,"developer_action","UNKNOWN")
        reason=getattr(gt,"developer_action_reasoning","") or ""
        prim,_=substantive_fix_buckets(reason)
        bucket=prim or ACTION_BUCKET.get(action)
        if bucket not in need: return None
        # extract logs
        try:
            tarball=B.tarball_path(run); excs=[]; samp=[]
            for fs in event.failed_steps[:3]:
                ex=extract_log_excerpt(zip_path=zip_path,tarball_name=tarball,
                                       job_file=fs.job_file,step_label=fs.step_label or "")
                excs.append(ex)
                for ml in ex.error_marker_lines[:2]:
                    c=ml.strip()[:120]
                    if c and c not in samp: samp.append(c)
            combined="\n".join(e.as_prompt_text() for e in excs)
            if not B.is_informative_log(combined): return None
            payload=[]
            for ex in excs:
                t=B._redact_secrets(ex.as_prompt_text())
                if len(t)>20000: t=t[:19999]+"…"
                payload.append({"job_file":getattr(ex,"job_file",""),"step_label":getattr(ex,"step_label",""),
                                "strategy":getattr(ex,"strategy_used",""),"text":t})
            return {"bucket":bucket,"case":{
                "case_label":f"{event.repo} — {event.commit_title[:60]}",
                "preflight_action":action,"target_bucket":bucket,"mined_topup":True,
                "gt_fix_window_days":int(win),
                "ground_truth":{"action":action,"method":method,"reasoning":reason},
                "intake":{"run_id":event.run_id,"repo":event.repo,"workflow":event.workflow,
                    "branch":event.branch,"event":event.event,"is_protected_branch":event.is_protected_branch,
                    "commit_sha":event.commit_sha,"commit_title":event.commit_title,"conclusion":event.conclusion,
                    "failure_detection":event.failure_detection,"failed_jobs_count":event.failed_jobs_count,
                    "n_jobs":event.n_jobs,"all_failures_are_tooling_artifacts":event.all_failures_are_tooling_artifacts},
                "extraction":{"total_steps_extracted":len(excs),"strategies":[e.strategy_used for e in excs],
                    "error_markers_found":sum(len(e.error_marker_lines) for e in excs),
                    "sample_error_lines":samp[:5],"log_excerpts":payload}}}
        except Exception:
            return None

    scanned=0; ctx=0; pending=[]
    def done(): return got["DEPENDENCY"]>=need["DEPENDENCY"] and got["CONFIG"]>=need["CONFIG"]

    def flush_batch(batch):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs={ex.submit(scrape_and_build,r):r for r in batch}
            for fu in as_completed(futs):
                res=fu.result()
                if not res: continue
                b=res["bucket"]
                with wlock:
                    if got.get(b,0)>=need[b]: continue
                    writer.write(res["case"]); got[b]+=1
                    print(f"  ✓ [{b}] {got[b]}/{need[b]}  {res['case']['intake']['repo']}",flush=True)

    with gzip.open(args.runs_path,"rt",encoding="utf-8") as f:
        for line in f:
            scanned+=1
            if scanned>args.scan_end or done(): break
            if scanned%5000==0:
                print(f"  scanned {scanned:,} | ctx {ctx} | DEP {got['DEPENDENCY']}/{need['DEPENDENCY']} CONFIG {got['CONFIG']}/{need['CONFIG']} | queued {len(pending)}",flush=True)
            try: run=json.loads(line)
            except: continue
            if not B.passes_context_gate(run,protected_only=True,min_steps_per_job=2,require_shell_step=True): continue
            meta=run.get("metadata") or {}; sha=(meta.get("head_commit") or {}).get("id","")
            owner,repo=B._parse_owner_repo(run.get("repository_name","")); started=B._run_started_at(meta); branch=meta.get("head_branch")
            if not(owner and repo and sha and branch and started): continue
            repo_full=run.get("repository_name","")
            rid=str((meta.get("id") or run.get("id") or ""))
            if rid in existing_runs or repo_full in existing_repos or repo_full in tried_repos: continue
            tried_repos.add(repo_full)
            # NO preflight here — phase 1 stays purely LOCAL (no network) so it can fill
            # batches fast. The full scrape (network) happens in parallel in flush_batch.
            ctx+=1
            pending.append(run)
            if len(pending)>=args.candidate_batch:
                flush_batch(pending); pending=[]
                if done(): break
    if pending and not done(): flush_batch(pending)
    writer.close()
    print(f"\nDone. {got} written. scanned={scanned:,} candidates_queued~={ctx}",flush=True)

if __name__=="__main__":
    main()
