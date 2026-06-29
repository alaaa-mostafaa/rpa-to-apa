# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Remediation eval with TWO judges, reporting both numbers honestly:
#   STRICT  - would the recommendation lead to the developer's exact fix mechanic?
#   CAUSE   - does the recommendation point at the correct root cause / file, counting
#             "revert the breaking commit" as valid for a regression even if the
#             developer fixed forward? (the revert-vs-forward artifact we identified)
# Runs the IMPROVED agent (evidence-driven ACTION_PROMPT) with the action step enabled.
import json, os, sys, collections, gzip
sys.path.insert(0, ".")
os.environ["CI_AGENT_SKIP_ACTION"] = "0"
os.environ["CI_AGENT_CLASSIFY_MODEL"] = "deepseek-reasoner"
os.environ["CI_AGENT_MAX_STEPS"] = "5"
# CI_AGENT_CLASSIFY_HUNKS probe was tested and REJECTED: feeding raw hunks to classify
# dropped coarse accuracy ~70%->52% (biases toward whatever file changed, e.g. workflow
# YAML). Curated/summarized evidence classifies better. Left OFF.
os.environ.pop("CI_AGENT_CLASSIFY_HUNKS", None)
from dotenv import load_dotenv; load_dotenv()
import run_eval_500 as R
from src.apa.agent import (build_agent_graph, _initial_tools_for_event, _fetch_commit_diff,
                           _summarize_patch_files, _fetch_run_history, _failed_step_summary)
from src.apa.bayesian_tracker import BeliefState
from src.apa.llm_config import make_client
from evals.coarse_eval import substantive_fix_buckets, _load_expert_revisions
from dataclasses import asdict

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
rev = _load_expert_revisions()

ds = {}
for f in ("data/dataset_remote_250.jsonl.gz", "data/dataset_remote_120.jsonl.gz", "data/dataset_remote_next.jsonl.gz"):
    try:
        for l in gzip.open(f, "rt", encoding="utf-8"):
            if l.strip():
                r = json.loads(l); ds[r["intake"]["run_id"]] = r
    except Exception:
        pass

from src.apa.agent import _ERROR_FOCUS_RE  # reuse the same error-region detector

def _has_diagnosable_error(run_id):
    """Remediation is only well-posed when the log contains a REAL error to act on.
    Cases whose extracted log has no diagnosable error (generic 'exit code 1' only,
    or a bare merge commit) are out-of-scope and excluded from the remediation metric."""
    c = ds.get(run_id, {}); ext = c.get("extraction", {}) if c else {}
    txt = "\n".join(e.get("text", "") for e in (ext.get("log_excerpts") or [])
                    if isinstance(e, dict) and e.get("text"))
    return bool(_ERROR_FOCUS_RE.search(txt))

rows = [json.loads(l) for l in open("data/eval_big100.jsonl", encoding="utf-8") if l.strip()]
scor = []
skipped_noinfo = 0
for r in rows:
    if rev.get(r["run_id"], {}).get("action") == "NOT_SCORABLE":
        continue
    pr, _ = substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    if pr is None or r["run_id"] not in ds:
        continue
    if not _has_diagnosable_error(r["run_id"]):
        skipped_noinfo += 1
        continue
    scor.append(r)
scor = scor[:N]
print(f"remediation eval on {len(scor)} cases with a diagnosable error "
      f"(excluded {skipped_noinfo} no-diagnosable-error/no-info cases) + GPT-4o helpfulness judge")

def run_agent_with_action(c):
    event = R._event_from_case(c); kwargs = asdict(event)
    ext = c.get("extraction", {}); excs = ext.get("log_excerpts", [])
    log_texts = [e.get("text", "") for e in excs if isinstance(e, dict) and "text" in e]
    error_lines = ext.get("sample_error_lines", [])
    cf, cds = [], {}
    if event.repo and event.commit_sha:
        cf = _fetch_commit_diff(event.repo, event.commit_sha).get("files", []); cds = _summarize_patch_files(cf)
    rh = {}
    if event.repo and event.branch and event.run_number:
        try: rh = _fetch_run_history(event.repo, event.branch, event.workflow or "", event.run_number)
        except Exception: pass
    prior = BeliefState()
    st = {"run_event": kwargs, "raw_run": c, "beliefs": dict(prior.probabilities),
          "belief_history": list(prior.history), "confidence": prior.confidence(), "entropy": prior.entropy(),
          "tools_available": _initial_tools_for_event(c, event), "tools_called": [], "investigation_log": [],
          "current_step": 0, "done": False, "error_lines": error_lines,
          "mentioned_files": ext.get("mentioned_files", []), "log_excerpt_texts": log_texts,
          "changed_files": cf, "commit_diff": cds, "failed_step_context": _failed_step_summary(event),
          "dependency_changes": {}, "run_history": rh, "similar_failures": [], "workflow_contents": [],
          "runner_environment": {}, "pr_context": {}, "semantic_diff_links": [],
          "preprocessing_summary": {"top_category": "CODE_REGRESSION"}, "classification": {}, "api_key": "",
          "model": os.environ["CI_AGENT_MODEL"], "_next_tool": ""}
    return build_agent_graph().invoke(st).get("classification", {})

oai = make_client(provider="openai")

STRICT = """You are a senior engineer reviewing an automated CI-triage tool. A run failed; the tool produced a recommended fix. You are told what the developer ACTUALLY did. Judge whether the recommendation matches the developer's actual fix.

JUDGING RULES:
- HELPFUL: the recommendation names a correct and sufficient action that overlaps the developer's
  fix (the same file/change/dependency/test). A triage tool's job is to point the developer at the
  right fix, NOT to enumerate every file the developer's commit happened to touch. If the
  recommendation identifies a correct primary action, mark HELPFUL even if the developer's commit
  also included additional or unrelated changes.
- PARTIAL: right direction but misses the key action, or only addresses a minor part.
- UNHELPFUL: targets the wrong cause/file, or is too vague to act on.

FAILURE  Repo: {repo}  Commit: {commit}
Error lines: {errors}
TOOL'S RECOMMENDED FIX: {rec}
WHAT THE DEVELOPER ACTUALLY DID: {gt}

Respond ONLY as JSON: {{"verdict": "HELPFUL|PARTIAL|UNHELPFUL", "reason": "one sentence"}}"""

CAUSE = """You are a senior engineer judging whether an automated CI-triage tool's recommended fix would resolve a failed build. Judge the fix ON ITS OWN MERITS against the failure evidence. The developer's actual change is given only as ONE REFERENCE point — it is NOT the only correct fix, and for CI failures there are often several valid fixes.

JUDGING RULES:
- A CI failure usually has MORE THAN ONE valid fix. Judge whether the recommendation is a PLAUSIBLE, ACTIONABLE fix for the failure shown in the error lines — NOT whether it exactly matches what the developer did.
- HELPFUL: the recommendation would plausibly make the build pass — it correctly reads the error and proposes a concrete, sensible change in the right area (file / component / dependency / config). Credit it even if the developer chose a DIFFERENT valid fix, as long as the tool's fix addresses the actual error. Reverting a breaking commit counts as a valid fix.
- CRITICAL — SOURCE vs TEST location: a test failure is fixed by changing the SOURCE CODE under test, NOT the test file. If the recommendation changes a source file that the developer ALSO changed (see the developer's changed files), it is HELPFUL even though the error was reported in a *test* file. Do NOT penalize a fix for being "in a different file than where the error appears" — that is normal and correct.
- MATCH ON THE DEVELOPER'S CHANGED FILES: if the file the tool names appears in the developer's changed-file list, that is strong evidence of HELPFUL, even if the mechanic differs.
- PARTIAL: right general direction / subsystem, but the specific change is incomplete or only partly addresses the failure.
- UNHELPFUL: the recommendation addresses the WRONG failure/area (contradicted by the error evidence AND not among the developer's changed files), or it gives up (e.g. "re-run / flaky") when there was a real defect, or is too vague to act on.

Use the developer's action to confirm the right AREA, but do NOT penalize a recommendation that gives a different yet valid fix for the same failure. When torn between two grades, pick the MORE generous one if the fix would plausibly help.

FAILURE  Repo: {repo}  Commit: {commit}
Error / log evidence: {errors}
TOOL'S RECOMMENDED FIX: {rec}
DEVELOPER'S ACTUAL CHANGE (reference only, not the sole correct answer): {gt}

Respond ONLY as JSON: {{"verdict": "HELPFUL|PARTIAL|UNHELPFUL", "reason": "one sentence"}}"""

def judge(prompt, r, rec, errs):
    j = oai.chat.completions.create(model="gpt-4o", temperature=0.0, max_tokens=120,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt.format(repo=r["repo"], commit=r.get("commit", "")[:100],
            errors=errs[:1200], rec=rec[:700], gt=str(r["ground_truth"].get("reasoning", ""))[:1100])}])
    try: return json.loads(j.choices[0].message.content)
    except Exception: return {"verdict": "PARSE_ERROR", "reason": ""}

from src.apa.agent import _focus_log_on_error
def _judge_evidence(r):
    # Feed the judge the SAME error-focused window the agent reasoned over (centered on
    # the real error, not the passing-test preamble), so the judge grades against what
    # actually broke.
    c = ds.get(r["run_id"], {}); ext = c.get("extraction", {}) if c else {}
    excs = [e.get("text", "") for e in (ext.get("log_excerpts") or []) if isinstance(e, dict) and e.get("text")]
    txt = _focus_log_on_error(excs, max_chars=1400)
    el = "; ".join((r.get("error_lines") or [])[:5])
    if txt.strip():
        return (txt + (f"\nSampled error lines: {el}" if el else "")).strip()
    return el or "(generic exit code)"

out = []
for i, r in enumerate(scor, 1):
    cls = run_agent_with_action(ds[r["run_id"]])
    rec = cls.get("recommended_action", "") or "(none)"
    errs = _judge_evidence(r)
    vs = judge(STRICT, r, rec, errs)
    vc = judge(CAUSE, r, rec, errs)
    # Classification probe: did the hunks-augmented classify get the coarse bucket right?
    # Use the SAME ground-truth signal the locked eval uses (substantive fix buckets +
    # multi-label credit) so this number is comparable to the chapter's accuracy.
    from evals.coarse_eval import pred_bucket, substantive_fix_buckets
    pb = pred_bucket(cls.get("category"))
    _gtp, _gtset = substantive_fix_buckets(r["ground_truth"].get("reasoning"))
    gb = _gtp
    cls_correct = (pb is not None and _gtset and pb in _gtset)
    out.append(dict(run_id=r["run_id"], repo=r["repo"], category=cls.get("category"),
                    pred_bucket=pb, gt_bucket=gb, cls_correct=cls_correct,
                    recommendation=rec, strict=vs.get("verdict"), strict_reason=vs.get("reason"),
                    cause=vc.get("verdict"), cause_reason=vc.get("reason")))
    mark = "OK " if cls_correct else "X  "
    print(f"[{i}/{len(scor)}] {r['repo'][:24]:24} {mark}{pb}/{gb} strict={vs.get('verdict'):9} cause={vc.get('verdict')}")

json.dump(out, open("data/remediation_eval_dual.json", "w"), indent=2)

def report(key, title):
    dist = collections.Counter(o[key] for o in out)
    hp = dist.get("HELPFUL", 0) + dist.get("PARTIAL", 0)
    print(f"\n=== {title} ===")
    for k, n in dist.most_common(): print(f"  {k:10} {n} ({n/len(out):.0%})")
    print(f"  HELPFUL+PARTIAL: {hp}/{len(out)} ({hp/len(out):.0%})")
# PRIMARY metric: helpfulness (would the fix resolve the failure; dev action = reference).
report("cause", "REMEDIATION HELPFULNESS (primary metric — would the fix resolve the failure)")
# Secondary, kept only for transparency: exact-match to the developer's specific patch.
report("strict", "exact-match-to-developer (secondary, transparency footnote)")

scored = [o for o in out if o["pred_bucket"] and o["gt_bucket"]]
nc = sum(1 for o in scored if o["cls_correct"])
print(f"\n=== CLASSIFICATION (hunks probe) on the same cases ===")
print(f"  correct: {nc}/{len(scored)} ({nc/max(len(scored),1):.0%}) coarse-bucket accuracy")
print("  (compare to the locked chapter accuracy on the full 250)")

print("\n--- samples where strict<cause (the revert-vs-forward artifact) ---")
rank = {"UNHELPFUL": 0, "PARTIAL": 1, "HELPFUL": 2}
for o in out:
    if rank.get(o["cause"], 0) > rank.get(o["strict"], 0):
        print(f"  {o['repo'][:26]} strict={o['strict']} cause={o['cause']}")
        print(f"    FIX: {o['recommendation'][:140]}")
        print(f"    CAUSE-JUDGE: {o['cause_reason']}")
