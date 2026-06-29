# CI/CD Failure Diagnosis — RPA vs. APA (thesis code and evaluation)

Research codebase for the thesis comparing **rule-based process automation (RPA)** with
**agentic process automation (APA)** for diagnosing failures in CI/CD pipelines. Three
diagnosis levels are implemented and evaluated:

- **L1 — RPA:** a deterministic signal battery over the failure log and run metadata.
- **L2 — APA:** a Bayesian belief tracker (nine failure categories) driving an LLM agent
  loop that plans evidence-gathering tools by expected information gain (EIG) and stops on
  an entropy threshold.
- **L3 — APA + retrieval:** L2 augmented with a retrieval prior over a store of past
  diagnoses.

Failures are classified into a nine-category taxonomy and mapped to four coarse families
(code / config / dependency / transient). Evaluation uses two tracks — agreement with the
developer's actual fix, and an LLM-judged defensibility check against the failure evidence
(GPT-4o as the primary judge plus a secondary LLM judge).

## Repository layout

- `src/apa/` — the diagnosis pipeline: intake parser, log extractor, Bayesian belief
  tracker, the EIG-planned agent loop, classification and remediation.
- `src/web/` — the FastAPI webhook server (GitHub Actions deployment path) and dashboard.
- `src/faf/` — a packaged adapter form of the agent (GitHub / Kubernetes adapters).
- `evals/` — core evaluation harness modules and LLM-judge utilities (importable).
- `evaluation/` — the runnable scripts, grouped by purpose. `run_eval_500.py` is the main
  two-track harness; subfolders hold the rest:
  - `judging/` — the LLM-judge and re-judge scripts (`fine_judge.py`, `rejudge_*.py`,
    `judge_l3*.py`, `dump_fullctx*.py`).
  - `retrieval/` — the L3 retrieval-prior evaluations (`l3_*.py`).
  - `remediation/` — the fix-recommendation evaluations (`l4_*.py`).
  - `analysis/` — agent-behaviour and quality analysis (`collect_traces.py`,
    `analyze_traces.py`, `actionability_eval.py`).
  - `dataset/` — dataset building / mining / validation.
  - `figures/` — figure generation. Run all scripts from the repository root (see below).
- `data/` — curated datasets (`dataset_remote_*.jsonl.gz`), the scored evaluation set
  (`eval_big100.jsonl`), agent traces and all result JSON files.
- `archive/` — `ground_truth_scraper.py`, the developer-fix ground-truth collector used by
  the evaluation harness.
- `images/`, `thesis_figures/` — generated figures and architecture diagrams.

## Running

Python 3.12. Install dependencies and provide the model keys via environment variables
(a local `.env` is read automatically and is **not** committed):

```bash
pip install -r requirements.txt

export DEEPSEEK_API_KEY=...   # diagnosis models (classify / planner / likelihood)
export OPENAI_API_KEY=...     # embeddings + GPT-4o judge (optional)
export GITHUB_TOKEN=...        # live commit-diff / run-history fetch (optional)
```

Run scripts **from the repository root** (so `src/`, `evals/` and `data/` resolve), with
`src` on the path (`PYTHONPATH=".;src"` on Windows, `PYTHONPATH=".:src"` on Unix):

```bash
python evaluation/run_eval_500.py                 # two-track RPA/APA evaluation
python evaluation/analysis/collect_traces.py 60   # instrumented agent-behaviour traces
python evaluation/analysis/analyze_traces.py      # tool-usage / depth / EIG analysis
```

## Data note

The raw failure-log corpora are **not included** in this repository, due to size and
third-party licensing:

- `data/runs_zenodo.json.gz` (~1 GB) — the raw GitHub Actions run corpus (available from
  the GHALogs / Zenodo dataset).
- `data/logchunks/` (~200 MB) — the LogChunks benchmark corpus (available from its authors).

The demo/comparison directories (`streaming_cases_10/`, `comparison_results/`) keep their
structured records (`run_full.json`, `run.json`, `metadata.json`) but the bulky raw log
dumps were trimmed to keep the repository small. Superseded/intermediate result files,
run logs and scratch artifacts have been removed; `data/` retains only the result files
the evaluation and analysis code actually uses.

All **derived, curated datasets** used by the evaluations (`data/*.jsonl.gz`) and every
**result file** are included, so the reported numbers can be inspected and re-judged without
the raw corpora.
