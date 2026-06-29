PLANNER_PROMPT = """You are investigating a CI/CD failure. Your goal is to determine what kind of failure this is.

Cheap deterministic facts are already available in the initial beliefs and investigation log:
branch type, protected branch status, job counts, failure detection mode, commit title/message, extracted error lines, and mentioned files.
Do not spend a tool call on facts that are already known.

CURRENT STATE:
  Step: {step}/{max_steps}
  Entropy: {entropy:.2f} bits
  Top beliefs: {top_beliefs}
  Tools already used: {tools_used}
  Tools still available: {tools_available}

EXPECTED INFORMATION GAIN PER TOOL (bits of entropy reduction, given current beliefs):
  A higher number means that tool is expected to resolve more uncertainty
  for a run with THIS belief profile. Use this as a calibrated prior.
{eig_table}

AVAILABLE TOOLS:
  deep_log_analysis      — Read the full failure excerpt and let the LLM reason over the raw error text. Best for root-cause diagnosis from logs.
  inspect_failed_step_context — Read failed-step metadata already in the run payload: runner, failed step label/type, detection mode.
  inspect_commit_diff    — Fetch the actual commit diff / changed files from GitHub. Best for telling code vs config vs dependency changes apart.
  inspect_dependency_changes — Focus only on dependency-manifest edits and version-bump evidence linked to the error text.
  inspect_runner_environment — Inspect runner images, workflow runtime pins, and toolchain/version mismatch clues.
  check_run_history      — Fetch recent workflow runs for the same branch/workflow. Best for spotting flakiness and recurring failures.
  inspect_workflow_file  — Fetch and parse changed workflow YAML files. Best for action pins, workflow syntax, and CI config drift.
  inspect_pr_context     — If the run belongs to a PR, inspect PR title, labels, and changed files for release-note / labeling / scope clues.
  search_similar_failures — APA-only semantic retrieval over prior failures. The cheap token-overlap retrieval was already applied in preprocessing.
  search_web_for_error   — Search StackOverflow and GitHub issues for obscure framework errors or missing packages.
  compare_previous_successful_log — Download the raw log of the last successful run and diff it against the current failed run to spot silent dependency bumps or infrastructure drift.

INVESTIGATION SO FAR:
{investigation_log}

DECISION:
If entropy is below {threshold:.2f} bits OR you've used most tools, choose "classify" to make your final judgment.
Otherwise, choose the tool that would give you the most useful information you don't have yet.
The EIG table is a strong prior — override it only if you have a domain-specific reason.

RULES:
1. You CANNOT call a tool that is already in "Tools already used" — those are done.
2. Choose from "Tools still available" only.
3. If you feel confident enough, choose "classify".
4. If "Tools still available" is empty, you MUST choose "classify".

Respond with ONLY a JSON object:
{{"tool": "tool_name_or_classify", "reasoning": "one sentence why"}}"""

CLASSIFY_PROMPT = """Based on your investigation, classify this CI/CD failure.

IMPORTANT: You are predicting the DEVELOPER'S FIX TYPE, not just the failure cause.
Ask yourself: "What did the developer have to change to fix this?" — then pick the category.

DETERMINISTIC PREPROCESSING:
{preprocessing_summary}

INVESTIGATION SUMMARY:
{investigation_log}

BAYESIAN BELIEFS (posterior after all evidence):
{beliefs}

ERROR EVIDENCE:
{error_lines}

CHANGED FILES IN TRIGGERING COMMIT:
{changed_files}

COMMIT DIFF SUMMARY:
{commit_diff}

FAILED STEP CONTEXT:
{failed_step_context}

DEPENDENCY CHANGE CONTEXT:
{dependency_changes}

RUNNER / ENVIRONMENT CONTEXT:
{runner_environment}

PR CONTEXT:
{pr_context}

WORKFLOW FILE SIGNALS:
{workflow_signals}

RECENT RUN HISTORY:
{run_history}

SIMILAR FAILURE RETRIEVAL:
{similar_failures}

FAILURE CONTEXT:
  Source: {source}
  Run ID: {run_id}
  Failed jobs: {failed}/{total}

Choose the MOST SPECIFIC category supported by the evidence.

CATEGORY DISCRIMINATION GUIDE:
  CODE_REGRESSION — a test or build failed because recently-changed source code introduced a bug or regression.
    DEFAULT to CODE_REGRESSION when source code files (.py, .java, .ts, .go, etc.) OR build scripts (CMakeLists.txt, setup.py, Makefile, .sh scripts) were changed in the triggering commit.
    NOT CODE_REGRESSION if: the error is "module not found", "version conflict", "deprecated action", "YAML parse error".
    NOT CODE_REGRESSION if: the error is a generic cancellation ("The operation was canceled") AND no source/build files were changed — then use CONFIG_ERROR.

  DEPENDENCY_CONFLICT — module-not-found, unresolved dependency, version incompatibility, pip/npm/cargo install failures.
    ALSO DEPENDENCY_CONFLICT if: a GitHub Action version is deprecated/unsupported, or a tool version is EOL.

  CONFIG_ERROR (= developer performed WORKFLOW_FIX) — the developer fixed a CI/workflow file (.yml, .github/, .circleci/).
    USE CONFIG_ERROR when: all jobs fail with "The operation was canceled" AND no source code or build scripts were changed in the triggering commit.

  TEST_FLAKINESS — test was already non-deterministic before this commit; no code or dependency change caused it.

  ENV_FLAKINESS — runner environment issue (disk space, network blip) not caused by code or config.

FIX-TYPE MAPPING (use this as your primary decision rule):
  Developer fixed source code files (.py, .js, .java, etc.) → CODE_REGRESSION
  Developer fixed a workflow file (.yml, .github/) → CONFIG_ERROR
  Developer pinned/upgraded a dependency (requirements.txt, package.json) → DEPENDENCY_CONFLICT
  Developer reverted the triggering commit → CODE_REGRESSION
  Developer fixed a test → TEST_FLAKINESS or CODE_REGRESSION

CATEGORIES: {categories}

Respond with ONLY a JSON object:
{{
  "reasoning": "2-3 sentences citing specific evidence. State what type of fix you believe the developer applied.",
  "category": "one category",
  "severity": "CRITICAL|HIGH|MODERATE|LOW",
  "confidence": 0.0-1.0
}}"""

DEVIL_ADVOCATE_PROMPT = """You are a peer reviewer challenging a CI/CD failure diagnosis.

Original diagnosis: {category} (confidence: {confidence:.0%})
Reasoning given: {reasoning}

Investigation evidence:
{investigation_log}

Bayesian beliefs (posterior):
{beliefs}

Error lines: {error_lines}

Your job: Play devil's advocate. Ask: "What if the diagnosis is WRONG?"
- What evidence *contradicts* or *is unexplained by* the original diagnosis?
- Is there a simpler or more specific alternative category that better fits the evidence?

If you find strong contradicting evidence, provide an alternative category.
If the original diagnosis is well-supported, confirm it.

Respond with ONLY a JSON object:
{{
  "upheld": true/false,
  "alternative_category": "category name if upheld=false, else null",
  "critique": "1-2 sentences on what evidence supports or contradicts the original diagnosis"
}}"""

ACTION_PROMPT = """Based on this CI/CD failure diagnosis, recommend a specific concrete fix action.

Diagnosis: {category} (confidence: {confidence:.0%})
Reasoning: {reasoning}
Error lines: {error_lines}

Write ONE specific, actionable fix recommendation (1-2 sentences). Be concrete:
- Name the specific file to change if possible
- Name the specific action (revert, pin version X, fix test Y, update action to v4)
- Do NOT say generic things like "investigate the issue"

Respond with ONLY a JSON object:
{{
  "recommended_action": "specific fix description"
}}"""
