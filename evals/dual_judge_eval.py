#!/usr/bin/env python3
"""
dual_judge_eval.py
==================
Score RPA-vs-APA under TWO complementary views, to separate "did the prediction
match the developer's fix" from "was the prediction the right diagnosis given the
evidence the classifier actually had".

  Judge A  (fix-based, deterministic): predicted bucket vs the developer-action
           bucket. Conservative; penalizes a correct diagnosis when the developer
           happened to fix the failure in a different layer. (= coarse_eval)

  Judge B  (evidence-based, GPT labeler): an independent GPT judge reads ONLY the
           triage evidence (log excerpt + triggering commit + changed files) and
           labels the failure category. BOTH systems are then scored against that
           single label, so the judge cannot favor whichever prediction it grades.
           If the evidence is uninformative it returns UNINFORMATIVE -> NOT_SCORABLE.

Honest caveat: Judge B measures agreement with a strong LLM's reading of the
evidence. Because APA is itself an LLM, this can tilt toward APA vs rule-based RPA.
Report BOTH: fix-based (conservative) brackets the low end, evidence-based the high.

Usage:
  python evals/dual_judge_eval.py [predictions.jsonl]   # default eval_t0_runA.jsonl
"""
from __future__ import annotations
import json, gzip, os, sys, collections
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

from evals.coarse_eval import (
    pred_bucket, gt_bucket, coarse_verdict, BUCKETS,
    substantive_fix_buckets, substantive_credit,
)
from src.apa.llm_config import make_client

PRED_FILE = sys.argv[1] if len(sys.argv) > 1 else "data/eval_t0_runA.jsonl"
DATASET = "data/dataset_diverse_500.jsonl.gz"
JUDGE_MODEL = os.environ.get("EVIDENCE_JUDGE_MODEL", "gpt-4o-mini")
CACHE = Path("data/evidence_labels_v2.json")

# Fine category -> coarse bucket (matches coarse_eval.PREDICTION_TO_COARSE)
FINE_TO_COARSE = {
    "CODE_REGRESSION": "CODE", "QUALITY_VIOLATION": "CODE", "TEST_FLAKINESS": "CODE",
    "DEPENDENCY_CONFLICT": "DEPENDENCY",
    "CONFIG_ERROR": "CONFIG", "INFRA_INCOMPATIBILITY": "CONFIG",
    "ENV_FLAKINESS": "TRANSIENT", "CASCADE_FAILURE": "TRANSIENT", "TOOLING_ARTIFACT": "TRANSIENT",
}

PROMPT = """You are labeling the TRUE failure category of a CI/CD run, using ONLY the
evidence available at triage time (the failure log, the triggering commit, and the
files it changed). Do NOT guess from how it was later fixed — you are not given that.

EVIDENCE
  Repo / workflow: {repo} / {workflow}
  Triggering commit: {commit}
  Files changed in the triggering commit: {changed}
  Files named in the error log: {mentioned}
  Error / log excerpt:
{log}

CATEGORIES (pick the single best-supported one):
  CODE_REGRESSION      - compile/build/test failure from changed source or build scripts
  QUALITY_VIOLATION    - a linter/static-analysis tool rejected the code (eslint, flake8, etc.)
  TEST_FLAKINESS       - a non-deterministic/intermittent test failure
  DEPENDENCY_CONFLICT  - missing module, version conflict, install/resolution failure, EOL/deprecated dep or action
  CONFIG_ERROR         - CI/workflow misconfig: bad YAML, wrong runner, missing secrets/keys, cancelled with no code error
  INFRA_INCOMPATIBILITY- runner/toolchain/glibc/system-library mismatch
  ENV_FLAKINESS        - transient network/rate-limit/runner outage (a retry would fix)
  CASCADE_FAILURE      - failed only because a dependency job/step failed first
  TOOLING_ARTIFACT     - failure is an artifact of the CI tooling, not the project

A failure can be genuinely AMBIGUOUS — the evidence may support more than one
category as a reasonable call. List the single best-supported category as
"evidence_category", and ALSO list any OTHER categories that are a defensible read
of this SAME evidence in "also_defensible".

DISCIPLINE (this keeps the judge honest, do not relax it):
  - Put a category in "also_defensible" ONLY if the log/commit shows POSITIVE
    evidence for it. If the evidence clearly points to ONE category, leave
    "also_defensible" empty.
  - NEVER list a category the evidence CONTRADICTS or merely "could happen in
    general". Plausible-in-the-abstract is NOT enough — it must be supported here.

CRITICAL — be VERY strict about this. You may assign a category ONLY if the log
contains an ACTUAL ERROR/FAILURE MESSAGE (a compile error, a failed assertion, a
stack trace, "module not found", a named lint violation, "unauthorized", a YAML
error, a timeout, etc.). The following are NOT diagnostics and on their own mean
the cause is UNDETERMINABLE — set evidence_category to "UNINFORMATIVE":
  - "Process completed with exit code 1/2/127", "The operation was canceled"
  - git checkout / clone / fetch / submodule output, "Reset branch", "new ref"
  - "Compiling ..." / build progress with no error, env/setup echoes
  - the COMMAND that was run (e.g. "Run pytest ...") with no failure output after it
Do NOT infer a category from the workflow's name, the repo, or what the step was
trying to do. No explicit error visible -> "UNINFORMATIVE". When in doubt, choose
UNINFORMATIVE.

Respond ONLY with JSON:
{{"evidence_category": "<one category or UNINFORMATIVE>", "also_defensible": ["<other evidence-supported categories, or empty>"], "key_signal": "<the log line or file that decides it>"}}"""


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_dataset():
    out = {}
    with gzip.open(DATASET, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["intake"]["run_id"]] = r
    return out


def build_evidence(case):
    ik = case.get("intake", {}); ext = case.get("extraction", {})
    excerpts = ext.get("log_excerpts", []) or []
    log = "\n".join(e.get("text", "") for e in excerpts)[:5000] or "(no log captured)"
    changed = ", ".join(f.get("filename", "") for f in []) or "(not available)"
    return {
        "repo": ik.get("repo", ""), "workflow": ik.get("workflow", ""),
        "commit": (ik.get("commit_title", "") or "")[:120],
        "changed": changed,
        "mentioned": ", ".join((ext.get("mentioned_files") or [])[:10]) or "(none)",
        "log": log,
    }


def evidence_label(client, ev) -> dict:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise CI-failure labeler. Output ONLY valid JSON."},
            {"role": "user", "content": PROMPT.format(**ev)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    primary = (data.get("evidence_category") or "UNINFORMATIVE").upper()
    defensible = [str(c).upper() for c in (data.get("also_defensible") or [])]
    return {"primary": primary, "defensible": defensible}


def main():
    preds = load_jsonl(PRED_FILE)
    ds = load_dataset()
    client = make_client(provider="openai")
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    ev_primary = {}   # run_id -> primary coarse bucket, or None if uninformative
    ev_allowed = {}   # run_id -> set of evidence-supported coarse buckets (primary + defensible)
    for i, r in enumerate(preds, 1):
        rid = r["run_id"]
        if rid in cache and isinstance(cache[rid], dict):
            lab = cache[rid]
        else:
            case = ds.get(rid, {})
            lab = evidence_label(client, build_evidence(case)) if case else {"primary": "UNINFORMATIVE", "defensible": []}
            cache[rid] = lab
            CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        pb = FINE_TO_COARSE.get(lab["primary"])
        ev_primary[rid] = pb
        allowed = {pb} if pb else set()
        for d in lab.get("defensible", []):
            db = FINE_TO_COARSE.get(d)
            if db:
                allowed.add(db)
        ev_allowed[rid] = allowed
        extra = ("+" + ",".join(sorted(allowed - {pb}))) if (allowed - {pb}) else ""
        print(f"  [{i}/{len(preds)}] {rid[:42]:42} evidence={lab['primary']}->{pb}{extra}")

    def cat(r, s): return (r.get(s, {}).get("prediction") or {}).get("category")

    # ---- Judge A: fix-based (substantive re-bucketing, partial credit) ----
    def tally_fix(exclude_dep=False):
        per = {b: [0.0, 0] for b in BUCKETS}
        for r in preds:
            primary, _ = substantive_fix_buckets(r["ground_truth"].get("reasoning"))
            if primary is None: continue
            if exclude_dep and primary == "DEPENDENCY": continue
            per[primary][1] += 1
            per[primary][0] += substantive_credit(cat(r, SIDE), r["ground_truth"].get("reasoning"), partial=True) or 0.0
        return per

    # ---- Judge B-strict: prediction must match the PRIMARY evidence cause ----
    def tally_ev_strict():
        per = {b: [0, 0] for b in BUCKETS}
        for r in preds:
            eb = ev_primary.get(r["run_id"])
            if eb is None: continue          # uninformative -> NOT_SCORABLE
            per[eb][1] += 1
            per[eb][0] += 1 if pred_bucket(cat(r, SIDE)) == eb else 0
        return per

    # ---- Judge B-lenient: prediction counts if it is ANY evidence-supported call ----
    def tally_ev_lenient():
        per = {b: [0, 0] for b in BUCKETS}
        for r in preds:
            allowed = ev_allowed.get(r["run_id"]) or set()
            if not allowed: continue         # uninformative -> NOT_SCORABLE
            eb = ev_primary[r["run_id"]]      # group by primary cause
            per[eb][1] += 1
            per[eb][0] += 1 if pred_bucket(cat(r, SIDE)) in allowed else 0
        return per

    def show(name, per):
        used = {b: v for b, v in per.items() if v[1]}
        tn = sum(v[1] for v in used.values()); tc = sum(v[0] for v in used.values())
        macro = sum(v[0] / v[1] for v in used.values()) / len(used) if used else 0
        cells = "  ".join(f"{b}:{v[0]}/{v[1]}" for b, v in used.items())
        print(f"   {name:24} micro {tc}/{tn}={tc/tn:.0%}  macro {macro:.0%}   [{cells}]")

    global SIDE
    print("\n================ DUAL-JUDGE RESULTS ================")
    n_uninf = sum(1 for v in ev_primary.values() if v is None)
    n_ambig = sum(1 for v in ev_allowed.values() if len(v) > 1)
    print(f"evidence-judge: {n_uninf}/{len(preds)} UNINFORMATIVE (dropped), "
          f"{n_ambig} ambiguous (>1 defensible category)\n")
    for SIDE in ("rpa", "apa"):
        print(f"-- {SIDE.upper()} --")
        show("Judge A fix (substantive)", tally_fix())
        show("Judge A fix (excl. dep)", tally_fix(exclude_dep=True))
        show("Judge B evidence (strict)", tally_ev_strict())
        show("Judge B evidence (right-call)", tally_ev_lenient())


if __name__ == "__main__":
    raise SystemExit(main())
