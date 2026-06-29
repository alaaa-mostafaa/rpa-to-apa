import json
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from faf.exceptions import CorpusNotFoundError

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "corpus" / "default_cases.json"
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent / "corpus" / "chroma_db"

def _case_to_text(commit_title: str, error_lines: List[str], mentioned_files: Optional[List] = None) -> str:
    """Builds a semantic string representation of a failure case for embedding."""
    errors = " | ".join(line.strip() for line in error_lines[:5] if line.strip())
    errors = errors or "no error text"
    title  = (commit_title or "").strip() or "no commit title"
    text   = f"commit: {title}  errors: {errors}"

    if mentioned_files:
        file_tokens: list[str] = []
        for item in mentioned_files[:8]:
            raw = item.get("path", "") if isinstance(item, dict) else str(item)
            basename = raw.replace("\\\\", "/").rstrip("/").rsplit("/", 1)[-1]
            if basename and len(basename) > 1:
                file_tokens.append(basename)
        if file_tokens:
            seen: set[str] = set()
            unique_tokens = [t for t in file_tokens if not (t in seen or seen.add(t))]
            text += "  files: " + " ".join(unique_tokens)
    return text

class ChromaCaseStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or str(DEFAULT_CHROMA_PATH)
        self._chroma = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(anonymized_telemetry=False),
        )
        # Using Chroma's default sentence-transformers model (runs locally, no API key needed)
        self._collection = self._chroma.get_or_create_collection(
            name="ci_failure_cases",
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def bootstrap_from_eval_file(self, corpus_path: Optional[Path] = None) -> int:
        """Populate the Chroma collection from a JSON corpus file."""
        cp = corpus_path or DEFAULT_CORPUS_PATH

        if not cp.exists():
            raise CorpusNotFoundError(f"Corpus file not found: {cp}")

        with cp.open("r", encoding="utf-8") as f:
            eval_results = json.load(f)

        existing_ids = set()
        if self._collection.count() > 0:
            all_meta = self._collection.get(include=[])
            existing_ids = set(all_meta.get("ids", []))

        inserted = 0
        for result in eval_results:
            intake = result.get("intake", {})
            cl = result.get("classification", {})
            gt = result.get("ground_truth", {})

            run_id = str(intake.get("run_id", ""))
            category = cl.get("category", "")

            if not run_id or not category or category == "UNKNOWN" or run_id in existing_ids:
                continue

            text = _case_to_text(
                commit_title=intake.get("commit_title", ""),
                error_lines=result.get("extraction", {}).get("sample_error_lines", []),
                mentioned_files=result.get("extraction", {}).get("mentioned_files", []),
            )

            self._collection.upsert(
                ids=[run_id],
                documents=[text],
                metadatas=[{
                    "run_id": run_id,
                    "repo": str(intake.get("repo", "")),
                    "workflow": str(intake.get("workflow", "")),
                    "category": category,
                    "gt_verdict": str(gt.get("match_verdict", "NO_DATA")),
                    "n_failed": int(intake.get("failed_jobs_count", 0)),
                    "n_total": int(intake.get("n_jobs", 1)),
                }],
            )
            inserted += 1
            existing_ids.add(run_id)
        
        return inserted

    def find_similar_case_records(self, commit_title: str, error_lines: List[str],
                                  mentioned_files: Optional[List] = None, k: int = 5, min_sim: float = 0.72) -> List[dict]:
        if self._collection.count() == 0:
            return []

        query_text = _case_to_text(
            commit_title=commit_title,
            error_lines=error_lines,
            mentioned_files=mentioned_files,
        )
        if not query_text.strip():
            return []

        n_results = min(k * 2, self._collection.count())
        if n_results == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
        except Exception:
            return []

        records = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            similarity = 1.0 - dist
            if similarity < min_sim:
                continue
            records.append({
                "run_id": meta.get("run_id", ""),
                "repo": meta.get("repo", ""),
                "workflow": meta.get("workflow", ""),
                "category": meta.get("category", ""),
                "similarity": round(similarity, 3),
                "gt_verdict": meta.get("gt_verdict", "NO_DATA"),
            })
            if len(records) >= k:
                break
        return records

_store: Optional[ChromaCaseStore] = None

def get_chroma_store(path: Optional[str] = None) -> ChromaCaseStore:
    global _store
    if _store is None:
        _store = ChromaCaseStore(path=path)
        if _store.count() == 0:
            _store.bootstrap_from_eval_file()
    return _store
