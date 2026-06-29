import json
import os
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from openai import OpenAI



from src.apa.llm_usage import record_usage


def _normalize_classifier_output(classifier_output: Any) -> Dict[str, Any]:
    if is_dataclass(classifier_output):
        return asdict(classifier_output)
    if isinstance(classifier_output, dict):
        return classifier_output
    return {
        "category": getattr(classifier_output, "category", None),
        "severity": getattr(classifier_output, "severity", None),
        "confidence": getattr(classifier_output, "confidence", None),
        "action": getattr(classifier_output, "action", None),
        "reasoning": getattr(classifier_output, "reasoning", None),
        "evidence": getattr(classifier_output, "evidence", None),
        "unknowns": getattr(classifier_output, "unknowns", None),
    }


def build_case_summary(
    event,
    sample_error_lines: List[str],
    log_excerpt: str = "",
    max_log_chars: int = 6000,
) -> str:
    """Compact case context for the judge.

    `log_excerpt` is the ACTUAL extracted failure log. It must be passed so the
    judge can do what its prompt instructs ("cross-check with the error log").
    Without it the judge only sees generic markers like
    "Process completed with exit code 1" and is forced to mark every
    log-dependent category (lint/flaky/infra/...) WRONG for lack of evidence —
    a systematic bias against any system that reads the log to make a finer call.
    """
    log_block = (log_excerpt or "").strip()
    if log_block and max_log_chars and len(log_block) > max_log_chars:
        # The extractor front-loads the failing step + "--- error context ---",
        # so keeping the head retains the diagnostic region.
        log_block = log_block[:max_log_chars] + "\n…[truncated]"
    log_section = log_block if log_block else "none extracted"
    return (
        f"Repo: {event.repo}\n"
        f"Workflow: {event.workflow}\n"
        f"Branch: {event.branch} (protected: {event.is_protected_branch})\n"
        f"Commit: {event.commit_sha} — {event.commit_title}\n"
        f"Event: {event.event}\n"
        f"Conclusion: {event.conclusion}\n"
        f"Failed jobs: {event.failed_jobs_count}/{event.n_jobs}\n"
        f"Failure detection: {event.failure_detection}\n"
        f"Key errors from logs: {'; '.join(sample_error_lines[:5]) or 'none extracted'}\n"
        f"\n--- Full extracted failure log ---\n{log_section}"
    )


def _keyword_tokens(text: Any) -> set[str]:
    if not text:
        return set()
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    return {
        tok
        for tok in re.findall(r"[a-zA-Z0-9_.-]{3,}", str(text).lower())
        if tok not in {"the", "and", "for", "with", "from", "that", "this", "not"}
    }


def _relevance_tokens(classifier: Dict[str, Any], ground_truth) -> set[str]:
    tokens = set()
    for value in (
        classifier.get("category"),
        classifier.get("action"),
        classifier.get("reasoning"),
        classifier.get("evidence"),
        classifier.get("unknowns"),
        getattr(ground_truth, "developer_action_reasoning", ""),
        getattr(ground_truth, "signals", {}),
    ):
        tokens |= _keyword_tokens(value)
    tokens |= {
        "workflow", "github", "action", "timeout", "retry", "assert",
        "test", "flake", "lint", "docker", "node", "python", "gradle",
        "cargo", "pytest", "dependency", "version", "build", "config",
    }
    return tokens


def _score_file_summary(file_summary: Dict[str, Any], relevance_tokens: set[str]) -> tuple[int, int]:
    filename = str(file_summary.get("filename", "")).lower()
    patch = str(file_summary.get("patch_excerpt", "")).lower()
    text = f"{filename}\n{patch}"
    token_hits = sum(1 for tok in relevance_tokens if tok in text)

    filename_bonus = 0
    if any(x in filename for x in (".github/", "workflow", "action.yml", "dockerfile", "compose", "build.gradle", "cargo.toml", "pyproject.toml", "package.json", "requirements", "go.mod", "pom.xml")):
        filename_bonus += 4
    if any(x in filename for x in ("test", "spec", "lint", "flake", "timeout", "retry")):
        filename_bonus += 2

    size = int(file_summary.get("changes", 0) or 0)
    return (token_hits + filename_bonus, size)


def _format_follow_up_commits(ground_truth, classifier: Dict[str, Any]) -> str:
    commits = getattr(ground_truth, "follow_up_commits", []) or []
    if not commits:
        return "No follow-up commit details available."

    relevance_tokens = _relevance_tokens(classifier, ground_truth)
    sections = []
    for i, commit in enumerate(commits[:3], 1):
        file_lines = []
        ranked_files = sorted(
            commit.file_summaries or [],
            key=lambda f: _score_file_summary(f, relevance_tokens),
            reverse=True,
        )
        for f in ranked_files[:6]:
            line = (
                f"- {f.get('filename', '?')} "
                f"[{f.get('status', '?')}] +{f.get('additions', 0)} -{f.get('deletions', 0)}"
            )
            patch = (f.get("patch_excerpt") or "").strip()
            if patch:
                patch = patch.replace("\r", "")[:280]
                line += f"\n  Patch excerpt:\n{patch}"
            file_lines.append(line)
        omitted = max(0, len(ranked_files) - len(ranked_files[:6]))
        if omitted:
            file_lines.append(f"- ... {omitted} more changed files omitted")
        files_text = "\n".join(file_lines) if file_lines else "- No file details available."
        sections.append(
            f"Commit {i}: {commit.sha}\n"
            f"  Date: {commit.date}\n"
            f"  Author: {commit.author}\n"
            f"  Title: {commit.message_title}\n"
            f"  Message: {commit.message_full[:400]}\n"
            f"  Total diff: +{commit.additions} -{commit.deletions}\n"
            f"  Files:\n{files_text}"
        )
    omitted_commits = max(0, len(commits) - len(commits[:3]))
    if omitted_commits:
        sections.append(f"... {omitted_commits} more follow-up commits omitted")
    return "\n\n".join(sections)


def _format_ground_truth_context(ground_truth, classifier: Dict[str, Any]) -> str:
    signals = getattr(ground_truth, "signals", {}) or {}
    try:
        signals_text = json.dumps(signals, ensure_ascii=False, sort_keys=True)
    except TypeError:
        signals_text = str(signals)

    return (
        f"Observed developer action: {ground_truth.developer_action}\n"
        f"Ground-truth method: {ground_truth.classification_method}\n"
        f"Ground-truth reasoning: {ground_truth.developer_action_reasoning}\n"
        f"Ground-truth signals:\n{signals_text}\n\n"
        f"Follow-up commit evidence:\n{_format_follow_up_commits(ground_truth, classifier)}"
    )


# Deterministic category->fix mapping. Each predicted category lists the
# developer-fix actions that count as a CORRECT match. Applied in code BEFORE
# the LLM so unambiguous cases get a consistent verdict (an LLM judge was
# observed flip-flopping on identical pairs like CODE_REGRESSION vs CODE_FIX).
# REVERT is intentionally absent: it is handled as NOT_SCORABLE before this
# table is consulted (a revert does not pin the failure category — see llm_judge).
#
# IMPORTANT — the ground truth is derived purely from the developer's CHANGED
# FILES + commit keywords, which is COARSER than the 9 prediction categories.
# File changes alone can confirm three categories uniquely:
#   CODE_REGRESSION     <- source/test edit
#   DEPENDENCY_CONFLICT <- dependency manifest pin/bump
#   CONFIG_ERROR        <- workflow/CI file edit
# But they CANNOT separate the "log-dependent" categories below — a lint fix, a
# flaky-test fix, and a logic-bug fix all look like "edited source = CODE_FIX",
# and an infra/workflow fix looks like WORKFLOW_FIX. For those, file evidence is
# necessary but NOT sufficient: only the ERROR LOG confirms them (e.g. ESLint
# output in the log -> QUALITY_VIOLATION). So these categories are NOT decided
# deterministically; they defer to the LLM judge, which reads the log.
_CATEGORY_FIX_MATCH = {
    "CODE_REGRESSION":       {"CODE_FIX", "CODE_CHANGE"},
    "DEPENDENCY_CONFLICT":   {"PIN_VERSION", "DEPENDENCY_CHANGE"},
    "CONFIG_ERROR":          {"WORKFLOW_FIX"},
}
# Log-dependent categories: the file-based ground truth cannot uniquely confirm
# these, so a deterministic CORRECT would be unfalsifiable. They always go to
# the LLM judge, which inspects the error log (lint output, flaky timeout,
# runner/toolchain mismatch, cascade) to decide.
_LOG_DEPENDENT_CATEGORIES = {
    "QUALITY_VIOLATION", "TEST_FLAKINESS", "INFRA_INCOMPATIBILITY",
    "ENV_FLAKINESS", "CASCADE_FAILURE", "TOOLING_ARTIFACT",
}
# Actions for which we trust a deterministic verdict. Anything outside this
# (e.g. unusual scraper labels) falls through to the LLM.
_KNOWN_ACTIONS = {
    "CODE_FIX", "CODE_CHANGE", "PIN_VERSION", "DEPENDENCY_CHANGE",
    "WORKFLOW_FIX", "RETRY",
}


def _deterministic_verdict(category: str, dev_action: str):
    """Return a verdict dict for clear-cut pairs, or None to defer to the LLM.

    Only the three file-distinguishable categories (CODE_REGRESSION,
    DEPENDENCY_CONFLICT, CONFIG_ERROR) get a deterministic verdict, because the
    developer's changed files uniquely confirm them. Log-dependent categories
    (lint, flakiness, infra, etc.) always return None so the LLM judge can read
    the error log — otherwise they would auto-pass against any matching file
    change and become unfalsifiable.
    """
    cat = (category or "").upper()
    act = (dev_action or "").upper()
    # Log-dependent categories must be confirmed by the LLM judge against the log.
    if cat in _LOG_DEPENDENT_CATEGORIES:
        return None
    if cat not in _CATEGORY_FIX_MATCH or act not in _KNOWN_ACTIONS:
        return None
    match_set = _CATEGORY_FIX_MATCH[cat]
    verdict = "CORRECT" if act in match_set else "WRONG"
    return {
        "verdict": verdict,
        "reasoning": f"Deterministic mapping: predicted {cat}; developer action {act}.",
        "confidence": 1.0,
        "evidence_used": [f"map:{cat}->{act}={verdict}"],
        "method": "deterministic",
    }


def llm_judge(case_summary: str, classifier_output: Any, ground_truth, client: OpenAI) -> Dict[str, Any]:
    classifier = _normalize_classifier_output(classifier_output)
    gt_context = _format_ground_truth_context(ground_truth, classifier)

    dev_action = getattr(ground_truth, "developer_action", "UNKNOWN")

    # If the ground truth shows the developer did something UNRELATED to
    # the failure, closed the PR, or the scraper itself failed,
    # we cannot score the AI's diagnosis — there is no reliable correct answer.
    # Not scorable: no usable developer fix to compare against. Note that
    # concrete merged-PR labels (CODE_FIX, WORKFLOW_FIX, PIN_VERSION,
    # CODE_CHANGE, DEPENDENCY_CHANGE from classify_pr_fix_type) ARE scorable
    # and deliberately fall through to the judge below. Only the opaque
    # PR_MERGED_NO_FILES / PR_MERGED_UNCLEAR variants are dropped.
    #
    # REVERT is NOT_SCORABLE: a revert only tells us the developer undid the
    # triggering commit, not WHAT KIND of failure it was. A reverted dependency
    # bump is causally a DEPENDENCY_CONFLICT; a reverted code change is a
    # CODE_REGRESSION — the same REVERT label maps to different failure
    # categories, so it cannot fairly score a category prediction. Excluding it
    # removes ambiguous cases rather than crediting a lossy mapping.
    if dev_action in ("UNRELATED", "PR_CLOSED_UNMERGED", "PR_MERGED",
                      "PR_MERGED_NO_FILES", "PR_MERGED_UNCLEAR", "REVERT",
                      "EVALUATION_ERROR", "UNKNOWN", "NO_IMMEDIATE_RESPONSE",
                      "NO_RELEVANT_RESPONSE", "NO_FOLLOW_UP"):
        return {
            "verdict": "NOT_SCORABLE",
            "reasoning": f"Ground truth action is {dev_action} — no reliable developer fix to compare against.",
            "confidence": 0.0,
            "evidence_used": [f"dev_action={dev_action}"]
        }

    # Decide the category<->action match DETERMINISTICALLY whenever the pair is
    # in our mapping table. This is critical for a FAIR RPA-vs-APA comparison:
    # the LLM judge was observed giving OPPOSITE verdicts to IDENTICAL
    # (category, action) pairs across the two systems — e.g. it marked
    # CODE_REGRESSION CORRECT against a REVERT for RPA but WRONG against the
    # same REVERT for APA, contradicting itself in a single sentence. The
    # category->fix mapping (CODE_REGRESSION<->{CODE_FIX,CODE_CHANGE,REVERT},
    # DEPENDENCY_CONFLICT<->{PIN_VERSION,...}, CONFIG_ERROR<->WORKFLOW_FIX) is
    # unambiguous, so a deterministic verdict removes that judge noise and makes
    # the same prediction score the same way no matter which system produced it.
    # Only pairs OUTSIDE the table (e.g. CASCADE_FAILURE, unusual labels) fall
    # through to the LLM judge below.
    det = _deterministic_verdict(classifier.get("category"), dev_action)
    if det is not None:
        return det

    prompt = f"""You are evaluating whether an AI correctly identified the failure category of a CI/CD run.

FAILURE LOG + CONTEXT:
{case_summary}

AI DIAGNOSIS:
  category: {classifier.get('category')}
  reasoning: {str(classifier.get('reasoning', ''))[:400]}

GROUND TRUTH — what the developer actually did to fix it:
{gt_context[:1500]}

YOUR TASK: Decide if the AI's failure category correctly describes what broke.

STEP 1 — Read the follow-up commit files above. What did the developer actually change?
  - If they changed source/test code (.py, .js, .ts, .java, .go, .rs, etc.) → the failure was CODE_REGRESSION or TEST_FLAKINESS
  - If they pinned/bumped a dependency (requirements.txt, package.json, Cargo.toml, go.mod, etc.) → DEPENDENCY_CONFLICT
  - If they edited a workflow/CI file (.yml in .github/, .circleci/, etc.) → CONFIG_ERROR or INFRA_INCOMPATIBILITY
  - If they reverted a prior commit → CODE_REGRESSION
  - If no code changed (just a retry or close) → ENV_FLAKINESS or TEST_FLAKINESS

STEP 2 — Cross-check with the error log. Does the log confirm that failure type?
  - Compile error / test assertion / runtime exception → CODE_REGRESSION
  - "module not found" / version conflict / install failure → DEPENDENCY_CONFLICT
  - Workflow YAML error / wrong action / cancelled with no code error → CONFIG_ERROR
  - Lint/style tool output with violations → QUALITY_VIOLATION
  - Flaky timeout that retry would fix → ENV_FLAKINESS or TEST_FLAKINESS

CRITICAL — these categories require POSITIVE evidence in the error log, because the
developer's changed files alone cannot confirm them (a lint fix and a logic-bug fix
both just "edit source code"):
  - QUALITY_VIOLATION → mark CORRECT only if the log actually shows lint/style/static-analysis output (ESLint, flake8, pylint, checkstyle, rubocop, etc.). If the log shows a normal compile/test failure instead, it is CODE_REGRESSION, so QUALITY_VIOLATION is WRONG.
  - TEST_FLAKINESS → mark CORRECT only if the failure looks non-deterministic (timeout, intermittent, passes on retry) AND no real code regression is shown. A deterministic test failure after a source change is CODE_REGRESSION, not TEST_FLAKINESS.
  - INFRA_INCOMPATIBILITY → mark CORRECT only if the log shows a runner/toolchain/glibc/version mismatch. A workflow edit alone is CONFIG_ERROR.
  - ENV_FLAKINESS → mark CORRECT only if the log shows a transient network/rate-limit/runner outage.
  - If the log is empty/uninformative ("exit code 1", "operation was canceled" with nothing else), you CANNOT confirm any of these log-dependent categories → mark WRONG.

STEP 3 — Verdict:
  CORRECT  → AI category matches what the developer fixed AND what the log shows
  WRONG    → AI category does not match (wrong failure type)
  PARTIAL  → ONLY if the commit genuinely touched two different fix types (e.g. both source code AND workflow files changed in the fix)
  NOT_SCORABLE → ground truth is too ambiguous to judge

Be strict: a plausible-but-wrong answer is WRONG. Default to WRONG when unsure.

Respond with ONLY this JSON:
{{"reasoning": "1-2 sentences: what did the developer fix, what does the log show, does the AI category match", "verdict": "CORRECT|PARTIAL|WRONG|NOT_SCORABLE", "confidence": 0.0-1.0, "evidence_used": ["key file or log line"]}}"""

    try:
        from src.apa.llm_usage import record_usage
        import os
        import time

        judge_model = os.environ.get("JUDGE_MODEL", "deepseek-chat")

        # The judge can run on a DIFFERENT provider than the system under test
        # (recommended: OpenAI gpt-4o-mini, so the evaluator is independent of
        # the DeepSeek model being evaluated). When JUDGE_PROVIDER is set, the
        # judge builds its own client and ignores the passed-in system client.
        judge_provider = os.environ.get("JUDGE_PROVIDER", "").lower()
        if judge_provider:
            from src.apa.llm_config import make_client as _mk
            client = _mk(provider=judge_provider)

        def _call_judge(model: str):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    from src.apa.llm_usage import usage_kwargs
                    # Reasoning models (deepseek-reasoner / r1) reject
                    # response_format=json_object, so omit it for them.
                    json_kw = ({} if any(t in model.lower() for t in ("reasoner", "r1", "thinking"))
                               else {"response_format": {"type": "json_object"}})
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a strict but fair evaluator of CI triage decisions. Output ONLY valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=600,
                        **json_kw,
                        **usage_kwargs(),
                    )
                    return resp
                except Exception as e:
                    err = str(e)
                    if ("429" in err or "Connection" in err or "timeout" in err.lower()) and attempt < max_retries - 1:
                        time.sleep(15 * (attempt + 1))
                    else:
                        raise e

        response = _call_judge(judge_model)
        record_usage(response, judge_model, call_type="chat", label="evaluation_judge.llm_judge")
        from src.apa.llm_usage import log_transcript
        log_transcript("judge", judge_model,
                       [{"role": "user", "content": prompt}], response,
                       extra={"category": classifier.get("category"),
                              "dev_action": dev_action})
        content = response.choices[0].message.content or ""
        # Strip DeepSeek <think>...</think> reasoning blocks before parsing
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'```(?:json)?', '', content).strip()
        if not content:
            # Fallback: retry once with deepseek-chat which never returns empty
            fallback_model = "deepseek-chat"
            response = _call_judge(fallback_model)
            record_usage(response, fallback_model, call_type="chat", label="evaluation_judge.llm_judge.fallback")
            content = response.choices[0].message.content or ""
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'```(?:json)?', '', content).strip()
        if not content:
            return {
                "verdict": "EVALUATION_ERROR",
                "reasoning": "Judge returned empty response even after fallback retry",
                "confidence": 0.0,
                "evidence_used": [],
            }
        data = json.loads(content)
        return {
            "verdict": data.get("verdict", "EVALUATION_ERROR"),
            "reasoning": data.get("reasoning", ""),
            "confidence": data.get("confidence", 0.0),
            "evidence_used": data.get("evidence_used", [])
        }
    except Exception as e:
        return {
            "verdict": "EVALUATION_ERROR",
            "reasoning": f"Judge failed: {e}",
            "confidence": 0.0,
            "evidence_used": [],
        }


def make_client() -> OpenAI:
    from llm_config import make_client as _make_client
    return _make_client()
