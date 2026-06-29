#!/usr/bin/env python3
"""
eval_extractor_quality.py
=========================
LLM-as-judge evaluation of the Log Extractor: does the deterministic
extraction + deduplication preserve the diagnostically important content?

For each case it runs the real extractor (extract_from_full_log) on the raw
log, then asks an OpenAI judge to score, on a 1-5 scale, how much of the
failure-defining signal survived in the extracted excerpt.

COST CONTROL (you have ~$1):
  The full raw logs total ~28M chars (~7M tokens) — judging against them in
  full would cost ~$17 on gpt-4o. Instead the judge receives only:
    - the extracted excerpt (<=20k chars, what the agent actually sees), and
    - a TAIL slice of the raw log (default 6k chars) as the ground-truth
      reference, since GitHub Actions error markers and the real failure
      cluster at the end of the relevant region.
  That is ~7k tokens/case. 20 cases:
    gpt-4o-mini : ~$0.02   (default)
    gpt-4o      : ~$0.36
  The script prints an upfront token/cost estimate and refuses to run if the
  projected cost exceeds --max-cost (default $0.90) unless --yes is passed.

Usage:
  export OPENAI_API_KEY=sk-...
  python eval_extractor_quality.py                      # gpt-4o-mini, all cases
  python eval_extractor_quality.py --model gpt-4o --n 20
"""

from __future__ import annotations

# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_5_cases import extract_from_full_log

# The agent never sees the raw extraction; it sees the 20k-capped excerpt
# (EXCERPT_FULL_MAX_CHARS in agent.py). Judge the same thing the agent judges,
# tail-biased because errors cluster at the end of the relevant region.
EXTRACTED_CAP = 20_000

def _cap_extracted(text: str) -> str:
    if len(text) <= EXTRACTED_CAP:
        return text
    return text[-EXTRACTED_CAP:]


def _build_extracted(full: str, label: str, semantic: bool, emb_client, emb_model: str):
    """Stage 1 (markers+dedup) and, if semantic=True, Stage 2 (refine on top).

    Returns (excerpt_text, strategy, n_markers, refine_info).
    """
    extracted, strategy, n_markers = extract_from_full_log(full, label)
    extracted = _cap_extracted(extracted)
    rinfo = {"applied": False}
    if semantic and emb_client is not None:
        from src.apa.semantic_refiner import refine_lines
        refined, rinfo = refine_lines(
            extracted.splitlines(), emb_client,
            embedding_model=emb_model, keep_max_lines=200,
        )
        if rinfo.get("applied"):
            extracted = "\n".join(refined)
    return extracted, strategy, n_markers, rinfo

# rough chars-per-token for cost estimation
CPT = 4.0
PRICE_IN = {   # USD per 1M input tokens
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
}
PRICE_OUT = {  # USD per 1M output tokens
    "gpt-4o-mini": 0.60,
    "gpt-4o": 10.00,
}

JUDGE_SYSTEM = (
    "You are evaluating a CI/CD log-extraction tool. You will see a TAIL slice "
    "of the raw failure log (the ground-truth signal region) and the tool's "
    "EXTRACTED excerpt. Judge how well the extracted excerpt preserves the "
    "information a engineer needs to diagnose the failure. Output ONLY JSON."
)

JUDGE_PROMPT = """RAW LOG (tail slice — the region where the failure manifests):
<<<RAW
{raw_tail}
RAW>>>

EXTRACTED EXCERPT (what the diagnostic agent actually receives):
<<<EXTRACTED
{extracted}
EXTRACTED>>>

Score how well the EXTRACTED excerpt preserves the diagnostically important
content from the raw log. Use this rubric:

  5 = all failure-defining lines present (the error/exception/exit marker and
      its immediate cause); a reader could diagnose from the excerpt alone.
  4 = the core error is present but some useful surrounding context is missing.
  3 = the error is identifiable but key context (which command/file/version)
      is missing or ambiguous.
  2 = only a generic failure indicator survived (e.g. "exit code 1") with no
      root-cause detail.
  1 = the failure-defining content was lost; the excerpt is misleading or empty.

Respond with ONLY this JSON object:
{{
  "score": 1-5,
  "key_signal_present": true/false,
  "missing": "one short phrase naming any important content dropped, or 'none'",
  "justification": "one sentence"
}}"""


def estimate_cost(cases, model, raw_tail_chars):
    in_tok = 0
    for c in cases:
        full = c.get("full_log", "")
        label = c.get("metadata", {}).get("label", "")
        extracted, _, _ = extract_from_full_log(full, label)
        extracted = _cap_extracted(extracted)
        raw_tail = full[-raw_tail_chars:]
        prompt = JUDGE_PROMPT.format(raw_tail=raw_tail, extracted=extracted)
        in_tok += (len(prompt) + len(JUDGE_SYSTEM)) / CPT
    out_tok = len(cases) * 120  # small JSON response
    cost = in_tok / 1e6 * PRICE_IN[model] + out_tok / 1e6 * PRICE_OUT[model]
    return int(in_tok), int(out_tok), cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/full_log_cases_20.json"))
    ap.add_argument("--model", default="gpt-4o-mini", choices=list(PRICE_IN))
    ap.add_argument("--n", type=int, default=0, help="limit number of cases (0 = all)")
    ap.add_argument("--raw-tail-chars", type=int, default=6000)
    ap.add_argument("--max-cost", type=float, default=0.90, help="abort if estimate exceeds this USD")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation gate")
    ap.add_argument("--out", type=Path, default=Path("data/extractor_quality_eval.json"))
    ap.add_argument("--semantic", action="store_true",
                    help="run Stage-2 semantic refinement on top of marker extraction")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="seconds between judge calls to avoid TPM rate limits")
    args = ap.parse_args()

    if args.semantic:
        # The refiner gates on this env var; the flag turns it on for this run.
        os.environ["CI_AGENT_SEMANTIC_REFINE"] = "1"
        if args.out == Path("data/extractor_quality_eval.json"):
            args.out = Path("data/extractor_quality_eval_semantic.json")

    cases = json.loads(args.data.read_text(encoding="utf-8"))
    if args.n:
        cases = cases[: args.n]

    in_tok, out_tok, cost = estimate_cost(cases, args.model, args.raw_tail_chars)
    print(f"Cases: {len(cases)} | model: {args.model}")
    print(f"Estimated input tokens: {in_tok:,} | output tokens: {out_tok:,}")
    print(f"Estimated cost: ${cost:.4f}  (cap ${args.max_cost:.2f})")

    if cost > args.max_cost and not args.yes:
        print(f"\nABORT: estimate ${cost:.4f} exceeds --max-cost ${args.max_cost:.2f}.")
        print("Re-run with a smaller --n, --model gpt-4o-mini, or --yes to override.")
        return 1

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set in environment.")
        return 1

    import time
    from openai import OpenAI
    client = OpenAI()
    emb_client = client if args.semantic else None

    print(f"Mode: {'Stage 1 + Stage 2 (semantic)' if args.semantic else 'Stage 1 only (markers+dedup)'}")

    def _judge(prompt: str) -> dict:
        """One judge call with up to 3 retries on transient errors (e.g. 429)."""
        last = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=200,
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:  # noqa: BLE001
                last = e
                wait = 4.0 * (attempt + 1)
                print(f"      retry {attempt+1}/3 after error: {str(e)[:80]} (waiting {wait:.0f}s)")
                time.sleep(wait)
        return {"score": None, "key_signal_present": None,
                "missing": "EVAL_ERROR", "justification": str(last)}

    results = []
    scores = []
    for i, c in enumerate(cases, 1):
        label = c.get("metadata", {}).get("label", c.get("case_label", f"case_{i}"))
        full = c.get("full_log", "")
        extracted, strategy, n_markers, rinfo = _build_extracted(
            full, label, args.semantic, emb_client, args.embedding_model)
        raw_tail = full[-args.raw_tail_chars:]
        prompt = JUDGE_PROMPT.format(raw_tail=raw_tail, extracted=extracted)
        data = _judge(prompt)
        if args.sleep:
            time.sleep(args.sleep)

        sc = data.get("score")
        if isinstance(sc, (int, float)):
            scores.append(sc)
        results.append({
            "case": label,
            "strategy": strategy,
            "error_markers": n_markers,
            "raw_chars": len(full),
            "extracted_chars": len(extracted),
            "semantic_applied": bool(rinfo.get("applied")),
            **data,
        })
        why = ""
        if data.get("key_signal_present") is False or (isinstance(sc, (int, float)) and sc <= 3):
            why = f"  -> missing: {data.get('missing','?')} | {data.get('justification','')[:90]}"
        sem_tag = " +sem" if rinfo.get("applied") else ""
        line = (f"  [{i:>2}/{len(cases)}] {label[:40]:40} "
                f"score={sc} signal={data.get('key_signal_present')} "
                f"({strategy}{sem_tag}){why}")
        print(line.encode("ascii", "replace").decode("ascii"))

    summary = {
        "model": args.model,
        "mode": "semantic" if args.semantic else "markers_only",
        "semantic_applied_count": sum(1 for r in results if r.get("semantic_applied")),
        "n_cases": len(cases),
        "n_scored": len(scores),
        "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        "score_distribution": {s: scores.count(s) for s in sorted(set(scores))},
        "key_signal_present_rate": round(
            sum(1 for r in results if r.get("key_signal_present") is True) / len(results), 2
        ) if results else None,
        "results": results,
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== EXTRACTOR QUALITY ===")
    print(f"mean score: {summary['mean_score']}/5 over {summary['n_scored']} cases")
    print(f"key signal present: {summary['key_signal_present_rate']:.0%}")
    print(f"distribution: {summary['score_distribution']}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
