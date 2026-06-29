#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evals/eval_chroma_retrieval.py
# ─────────────────────────────────────────────────────────────────────
# Retrieval quality benchmark: APA-v (ChromaDB) vs APA-t (tokenized).
#
# Methodology: leave-one-out cross-validation over honest_eval_results.json.
# For each case:
#   1. Bootstrap a Chroma collection that EXCLUDES the query case.
#   2. Query the collection with the held-out case's text.
#   3. Measure whether the top-k results agree with the true category.
#
# No LLM calls are made during evaluation — only the embedding API is
# used (but skipped if the run_id is already in retrieval_cache.json).
#
# Metrics:
#   - Top-1 category precision: does the top result match true category?
#   - Top-3 category accuracy: does any of top-3 results match?
#   - Mean reciprocal rank (MRR) for category match
#   - Coverage: fraction of queries with at least one result above threshold
#   - Mean similarity of returned results
#
# Usage:
#   python evals/eval_chroma_retrieval.py                    # slim mode
#   CHROMA_TEXT_MODE=rich python evals/eval_chroma_retrieval.py
#   python evals/eval_chroma_retrieval.py --both             # compare both modes
#   python evals/eval_chroma_retrieval.py --vs-tokenized     # include tokenized baseline
# ─────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import math
import re
import tempfile
import shutil
from collections import Counter
from pathlib import Path

# Force UTF-8 output on Windows so Unicode chars print cleanly
import io as _io
import sys as _sys
if hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_EVAL_PATH  = _DATA_DIR / "large_eval_results.json"
DEFAULT_CACHE_PATH = _DATA_DIR / "retrieval_cache.json"


# ─── metrics ─────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    """
    Each result entry:
      {
        "run_id": str, "true_category": str,
        "retrieved": [{"category": str, "similarity": float}, ...]
      }
    """
    n = len(results)
    if not n:
        return {}

    top1_match = 0
    topk_match = 0
    mrr_sum = 0.0
    coverage = 0
    sims = []

    for res in results:
        true_cat = res["true_category"]
        retrieved = res["retrieved"]

        if retrieved:
            coverage += 1
            sims.extend(r["similarity"] for r in retrieved)

        # Top-1 precision
        if retrieved and retrieved[0]["category"] == true_cat:
            top1_match += 1

        # Top-k accuracy & MRR
        found_at = None
        for rank, r in enumerate(retrieved[:5], start=1):
            if r["category"] == true_cat:
                topk_match += 1
                found_at = rank
                break
        if found_at:
            mrr_sum += 1.0 / found_at

    return {
        "n_queries": n,
        "coverage_pct": round(coverage / n * 100, 1),
        "top1_precision": round(top1_match / n * 100, 1),
        "topk_accuracy_k5": round(topk_match / n * 100, 1),
        "mrr": round(mrr_sum / n, 3),
        "mean_similarity": round(sum(sims) / len(sims), 3) if sims else 0.0,
    }


def print_metrics(label: str, metrics: dict) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Queries evaluated  : {metrics.get('n_queries', 0)}")
    print(f"  Coverage (>=1 hit) : {metrics.get('coverage_pct', 0)}%")
    print(f"  Top-1 precision    : {metrics.get('top1_precision', 0)}%")
    print(f"  Top-5 accuracy     : {metrics.get('topk_accuracy_k5', 0)}%")
    print(f"  MRR                : {metrics.get('mrr', 0):.3f}")
    print(f"  Mean similarity    : {metrics.get('mean_similarity', 0):.3f}")


# ─── leave-one-out ChromaDB evaluation ───────────────────────────────

def run_chroma_loo(
    eval_results: list,
    chroma_path: str,
    verbose: bool = False,
) -> list:
    """
    Leave-one-out evaluation using ChromaDB.

    For each entry in eval_results:
      1. Create a temporary Chroma collection containing all OTHER entries.
      2. Query it with the held-out entry.
      3. Record results.

    Returns a list of per-query result dicts.
    """
    import chromadb
    from chromadb.config import Settings

    # Import after ensuring the project root is on sys.path
    from src.apa.chroma_case_store import _case_to_text
    from src.apa.llm_config import make_client

    client = make_client()

    all_results = []
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_chroma_path = str(Path(tmpdir) / "chroma_loo")
        
        chroma_client = chromadb.PersistentClient(
            path=tmp_chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        coll_name = "ci_failure_cases_loo"
        collection = chroma_client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )

    print(f"  [loo] Upserting {len(eval_results)} documents into temp collection...")
    ids, documents, metadatas = [], [], []
    for entry in eval_results:
        intake = entry.get("intake", {})
        cl = entry.get("classification", {})
        ext = entry.get("extraction", {})
        run_id = str(intake.get("run_id", ""))
        category = cl.get("category", "")

        text = _case_to_text(
            commit_title=intake.get("commit_title", ""),
            error_lines=ext.get("sample_error_lines", []),
            mentioned_files=ext.get("mentioned_files", []),
        )

        ids.append(run_id)
        documents.append(text)
        metadatas.append({
            "run_id": run_id,
            "category": category,
            "repo": str(intake.get("repo", "")),
        })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # 3. Query leave-one-out
    print(f"  [loo] Running {len(eval_results)} queries...")
    for idx, query_entry in enumerate(eval_results):
        q_intake = query_entry.get("intake", {})
        q_category = query_entry.get("classification", {}).get("category", "")
        q_run_id = str(q_intake.get("run_id", ""))
        
        if verbose:
            print(f"    query {idx+1}/{len(eval_results)}: {q_run_id}")

        q_text = _case_to_text(
            commit_title=q_intake.get("commit_title", ""),
            error_lines=query_entry.get("extraction", {}).get("sample_error_lines", []),
            mentioned_files=query_entry.get("extraction", {}).get("mentioned_files", []),
        )

        # Query excluding the query document itself
        n_res = min(5, collection.count() - 1)
        if n_res <= 0:
            all_results.append({"run_id": q_run_id, "true_category": q_category, "retrieved": []})
            continue

        raw = collection.query(
            query_texts=[q_text],
            n_results=n_res,
            where={"run_id": {"$ne": q_run_id}},
            include=["metadatas", "distances"],
        )

        retrieved = []
        if raw.get("metadatas") and raw["metadatas"][0]:
            for meta, dist in zip(raw["metadatas"][0], raw["distances"][0]):
                sim = 1.0 - dist
                retrieved.append({
                    "category": meta.get("category", ""),
                    "run_id": meta.get("run_id", ""),
                    "similarity": round(sim, 3),
                })

        all_results.append({
            "run_id": q_run_id,
            "true_category": q_category,
            "retrieved": retrieved,
        })

    return all_results



# ─── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ChromaDB retrieval quality (leave-one-out)"
    )
    parser.add_argument("--eval",  default=str(DEFAULT_EVAL_PATH),  help="Eval results JSON")
    parser.add_argument("--chroma-path", default=str(_DATA_DIR / "chroma"), help="Chroma directory")
    parser.add_argument("--verbose",       action="store_true", help="Per-query progress")
    parser.add_argument("--save",          default="",          help="Save per-query results to this JSON file")
    args = parser.parse_args()

    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb is not installed. Run: pip install chromadb>=0.4.0")
        sys.exit(1)

    eval_path = Path(args.eval)
    if not eval_path.exists():
        print(f"ERROR: eval file not found: {eval_path}")
        sys.exit(1)

    with eval_path.open("r", encoding="utf-8") as f:
        eval_results = json.load(f)

    valid = [r for r in eval_results
             if r.get("intake", {}).get("run_id")
             and r.get("classification", {}).get("category")
             and r["classification"]["category"] != "UNKNOWN"]
    print(f"Eval corpus: {len(valid)} valid cases from {eval_path.name}")

    results = run_chroma_loo(valid, args.chroma_path, args.verbose)
    metrics = compute_metrics(results)
    print_metrics(f"ChromaDB ANN", metrics)
    
    # Save per-query results
    if args.save:
        save_path = Path(args.save)
        all_eval_results = {"metrics": metrics, "queries": results}
        save_path.write_text(json.dumps(all_eval_results, indent=2), encoding="utf-8")
        print(f"\n  Per-query results saved to: {save_path}")


if __name__ == "__main__":
    main()
