#!/usr/bin/env python3
# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# bootstrap_chroma.py
# ─────────────────────────────────────────────────────────────────────
# One-shot script to build the ChromaDB collection from your existing
# honest_eval_results.json.
#
# Run this once before the first USE_CHROMA=1 agent run:
#
#   python bootstrap_chroma.py
#
# Options:
#   --path PATH       Override Chroma directory (default: data/chroma/)
#   --eval  FILE      Override eval results file (default: data/large_eval_results.json)

#   --dry-run         Print stats without inserting
#   --force           Re-insert even if collection already has data
#
# The script reuses pre-computed embeddings from retrieval_cache.json,
# so it only calls the OpenAI embedding API for cases not yet cached.
# ─────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# ─── defaults ────────────────────────────────────────────────────────

_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_CHROMA_PATH = str(os.environ.get("CHROMA_PATH", _DATA_DIR / "chroma"))
DEFAULT_EVAL_PATH   = str(_DATA_DIR / "large_eval_results.json")



def _check_dependencies() -> None:
    """Fail fast with a clear message if chromadb is not installed."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        print("ERROR: chromadb is not installed.")
        print("       Run: pip install chromadb>=0.4.0")
        sys.exit(1)


def _print_collection_summary(store) -> None:
    n = store.count()
    print(f"\n  Collection '{store._collection.name}': {n} document(s)")
    if n > 0:
        # Sample a few entries for a quick sanity check
        sample = store._collection.get(limit=3, include=["metadatas"])
        for meta in sample.get("metadatas", []):
            print(
                f"    run_id={meta.get('run_id', '?')[:20]:22s} "
                f"category={meta.get('category', '?'):25s} "
                f"repo={meta.get('repo', '?')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap the ChromaDB case store from honest_eval_results.json"
    )
    parser.add_argument("--path",    default=DEFAULT_CHROMA_PATH, help="Chroma directory path")
    parser.add_argument("--eval",    default=DEFAULT_EVAL_PATH,   help="Eval results JSON file")
    parser.add_argument("--dry-run", action="store_true",         help="Print stats only, do not insert")
    parser.add_argument("--force",   action="store_true",         help="Re-bootstrap even if collection has data")
    args = parser.parse_args()

    _check_dependencies()

    eval_path  = Path(args.eval)
    chroma_path = args.path

    print(f"Chroma directory : {chroma_path}")
    print(f"Collection       : ci_failure_cases")
    print(f"Eval file        : {eval_path} {'(exists)' if eval_path.exists() else '(MISSING)'}")

    if not eval_path.exists():
        print(f"\nERROR: eval file not found: {eval_path}")
        sys.exit(1)

    # Count cases in the eval file
    with eval_path.open("r", encoding="utf-8") as f:
        eval_results = json.load(f)
    valid = [
        r for r in eval_results
        if r.get("intake", {}).get("run_id")
        and r.get("classification", {}).get("category")
        and r.get("classification", {}).get("category") != "UNKNOWN"
    ]
    print(f"\nEval file contains {len(eval_results)} entries, {len(valid)} valid (have run_id + category).")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return

    # Build the client
    from src.apa.llm_config import make_client
    client = make_client()

    # Initialise the store
    from src.apa.chroma_case_store import ChromaCaseStore
    store = ChromaCaseStore(path=chroma_path)

    existing_count = store.count()
    print(f"\nCollection currently has {existing_count} document(s).")

    if existing_count > 0 and not args.force:
        print(
            f"  Collection is not empty. Use --force to re-bootstrap.\n"
            f"  (New cases in the eval file are always added — idempotent upsert.)"
        )

    print("\nBootstrapping ...")
    inserted = store.bootstrap_from_eval_file(
        eval_path=eval_path,
        verbose=True,
    )

    print(f"\nDone. Inserted {inserted} new case(s).")
    _print_collection_summary(store)

    print(
        "\nTo activate ChromaDB retrieval in the agent, add to your .env:\n"
        "  USE_CHROMA=1\n"
        f"  CHROMA_PATH={chroma_path}\n"
    )


if __name__ == "__main__":
    main()
