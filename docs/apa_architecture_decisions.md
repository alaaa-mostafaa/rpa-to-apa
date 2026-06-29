# APA Architecture: Design Decisions & Rationale

A reference for the thesis presentation. Captures what the APA system looked like
before this redesign, what it looks like now, and the reasoning behind each change.
Every decision follows from one principle, stated once and applied repeatedly.

---

## The organizing principle

> **Evidence that is compact, free to fetch, and always available enters the belief
> state exactly once, through the single mandatory analysis step. The planner's tools
> are reserved for evidence that is deep, expensive, external, or only conditionally
> present.**

Two consequences fall out of this:

1. **Each fact updates the Bayesian posterior exactly once.** Reading the same evidence
   twice (two likelihood updates from one source) double-counts it and violates the
   conditional-independence assumption of the sequential Bayesian update. This is the
   single most important correctness property of the system.
2. **The planner's budget is spent only where value varies case-to-case.** Anything that
   can be gathered once for free is gathered once for free, not left to planner luck.

---

## Before → After at a glance

| Aspect | Before | After |
|---|---|---|
| Pre-loop LLM reads | `quick_log_scan` + `deep_log_analysis` (log read twice) | one mandatory `deep_log_analysis` (log read once) |
| Run history | fetched twice (preprocessing + optional `check_run_history` tool) | fetched once in preprocessing, rendered into header |
| Commit diff | optional `inspect_commit_diff` tool (paid LLM re-read) | prefetched free, families rendered into header |
| Failed-step shape | `inspect_failed_step_context` tool (paid LLM re-read) | rendered into header (zero LLM) |
| Metadata→APA | informal, inside prompts | serialized run-context header, one joint likelihood call |
| APA prior seeding | inherited RPA **posterior** (biased) | shared informed prior `P0(c)` (unbiased) |
| Disambiguation rules in likelihood prompt | hard-coded `if pattern then category` table | removed (taxonomy only; LLM reasons) |
| Tools that silently no-op'd beliefs | `search_web_for_error`, `compare_previous_successful_log` | both now do real framed likelihood updates |
| Selectable tools | ~10 | 7 (6 general + 1 conditional PR probe) |
| Per-tool prompt | one shared template for all | shared template + per-channel evidence frame |
| Step cap `K_max` | 7 (≈ tool count, cap never binds) | 5 (below tool count, forces prioritization) |
| Memory system | fingerprint cache + ChromaDB (confused) | ChromaDB only |

---

## The mandatory step (1 LLM call, before the planner loop)

`initialize → deep_log_analysis → planner`

The observation is a **run-context header** followed by the **full deduplicated log
excerpt** (≤20k chars / 350 lines), submitted in one call returning one likelihood vector.

The header carries every compact/free/always-available fact:
- branch + protected/bot flags
- trigger event
- run attempt (a retry that failed again = evidence against transient flakiness)
- actor + bot flag
- commit title
- job fan-out (e.g. 8 of 9 failed)
- first failed step (label, position, duration) + other failed steps
- failure detection mode (glossed into plain English)
- workflow path
- changed-file families (from the prefetched diff)
- recent failure rate (e.g. 3 of 5 recent runs failed)
- parent-run outcome (passed = this commit caused it; failed = predates it)

This step is the **LLM-based mirror of the RPA signal battery**: identical facts, but
one joint LLM likelihood instead of nine hand-coded probability tables. That symmetry is
the core of the thesis comparison — same evidence, rules vs. reasoning.

---

## The 7 selectable tools (and why each survives the principle)

| Tool | Why it is a tool, not a header fact |
|---|---|
| inspect_dependency_changes | deep: manifest patches + semantic version↔error matching |
| inspect_workflow_file | deep: fetches and parses YAML contents |
| inspect_runner_environment | deep: env images / pins / runtime versions |
| compare_previous_successful_log | expensive: downloads two full log archives |
| search_similar_failures | external: ChromaDB retrieval |
| search_web_for_error | external: web search |
| inspect_pr_context | conditional: only exists for PR-linked runs |

Each tool now prepends a **channel-specific evidence frame** before the shared likelihood
call — a natural-language statement of what that channel can and cannot discriminate
(its `P(o|c)` semantics). No pattern-to-category rules; that would re-import the RPA
tables. This gives every node a distinct prompt without duplicating the estimator.

---

## Why each thing was removed — the reasoning for the talk

### `quick_log_scan` (deleted)
Read the same log excerpt as `deep_log_analysis`, then ran a *separate* likelihood update.
Two updates from one dependent source → double-counting → posterior inflated toward the
log. Fix: one mandatory full read, one update.

### `inspect_commit_diff` (deleted as a tool)
The diff is already fetched for free in preprocessing and its families are already in the
header. A planner call would take that same family breakdown and run a *second* paid LLM
likelihood on it — a redundant re-read that double-counts. The diff is still important; it
just enters once, in the header, not twice.

### `inspect_failed_step_context` (deleted as a tool)
Step label/position/duration/detection-mode are compact, free, and always available — they
belong in the header by the principle. The tool was a paid re-read of header facts. Its
dossier is still built deterministically (zero LLM) for the final classify prompt.

### `check_run_history` (deleted as a tool)
Run history was *already* fetched in preprocessing to feed the RPA parent-commit signal.
The tool re-hit the same GitHub endpoint and applied the *hand-coded RPA tables* inside
APA — RPA logic hiding in the agent. Now the two facts (recent failure rate + parent
outcome) go in the header as English sentences for the LLM to weigh. Result: run-history
evidence is now *guaranteed* (was optional) and *LLM-interpreted* (was rule-interpreted),
while *saving* an API call.

### Disambiguation rules in the likelihood prompt (deleted)
The `if Cargo.toml-touched-but-no-version-error then CODE_REGRESSION` style rules were
hand-coded pattern→category mappings — the exact RPA approach APA is meant to transcend.
Keeping them in APA's prompt undercut the thesis claim. Removed; the prompt now gives only
the taxonomy and lets the model do the discrimination. (Note: in-code disambiguation notes
in `deep_log_analysis` were left in place under the "prompt-only" decision; flag if a
reviewer reads the implementation.)

### Fingerprint cache (deleted)
Two memory systems coexisted (SQLite fingerprint cache + ChromaDB). Confusing and
redundant. ChromaDB is the real runtime system; fingerprint code was live but
conceptually unused. Removed entirely.

---

## Why the silent no-op tools were fixed

`search_web_for_error` and `compare_previous_successful_log` previously called
`bs.update({}, ...)` — an empty likelihood, which the update rule turns into a uniform
vector, i.e. **no change to the posterior**. They gathered evidence that never entered the
belief state, only the investigation log. Now both route their findings through the shared
estimator with their own evidence frame, so every tool that runs performs a real update.
A failed/empty fetch correctly applies no update. (Also fixed a latent bug: snippets were
joined with a literal `\n` two-char string instead of newlines.)

---

## Why APA seeds from the prior, not the RPA posterior

APA must **not** inherit `P_final^RPA(c)` (RPA's conclusion after its signals run).
Seeding the agent with RPA's answer would bias every LLM step toward that answer before
the agent examines any evidence of its own — confounding the comparison. Both systems
share the same starting distribution `P0(c)`; APA re-derives its own posterior from
scratch. (The eval script had a bug here — it was seeding APA from the RPA tracker
posterior — now fixed.)

---

## Why `K_max = 5` (not 7, not 4)

- **Not 7:** with 7 selectable tools, a cap of 7 lets the agent run *everything* — the cap
  never binds, the "forces prioritization" argument is dead.
- **5 (chosen):** sits below the tool count, so the agent provably cannot run every tool
  and must prioritize — which is exactly what the EIG ranking is for. 5 tools cover every
  failure category (each category has a decisive channel within a few high-EIG calls); the
  cap excludes only the two weakest, corroborative probes (web search, PR context).
- **Not 4:** leaves zero slack on genuinely hard cases, which need ~3–4 substantive tools
  plus one recovery step when a tiebreaker surfaces a contradiction. The hardest cases are
  exactly where APA must out-investigate RPA, so throttling there is the wrong place to
  save one cheap call. Cost is not the binding constraint (model tiering handles that);
  correctness is.
- Caveat: reasoned from architecture, not yet measured. If the eval shows the cap rarely
  binds (entropy halts most cases at 2–3 steps), 4 vs 5 barely matters and 5 is the safer
  default. If many cases hit the cap, that signals the entropy threshold is too strict,
  not that the cap is wrong.

---

## Why `θ_stop = 1.0 bit`

1.0 bit = the entropy of a single fair binary choice — at most one yes/no question from
certainty. For the 9-class taxonomy that corresponds to the top category holding ≈86% of
the probability mass. Bracketing: 0.5 bit (≈94% mass) is stricter than LLM likelihoods
usually reach, so cases would run to the step cap without changing the answer; 1.5 bits
(≈76% mass) terminates with a quarter of the mass still on competitors. 1.0 sits in the
defensible middle.

(Max entropy for 9 classes = log2(9) ≈ 3.17 bits.)

---

## Model tiering (FrugalGPT-style)

- **Cheap model** (`CI_AGENT_MODEL`): the frequent calls — mandatory likelihood, each
  tool's likelihood, the planner (one per loop step).
- **Stronger model** (`CI_AGENT_CLASSIFY_MODEL`): the three single-call terminal prompts —
  classify, devil's advocate, recommended action.

Frequent work runs cheap; the final high-stakes judgment runs strong. Predictable per-case
cost ceiling, which is required for a fair cost comparison against the deterministic RPA
baseline.

---

## One-sentence summaries for slides

- **Principle:** cheap+universal evidence → one mandatory read; deep/expensive/conditional
  evidence → planner tools; every fact updates the posterior exactly once.
- **RPA vs APA:** identical evidence base, identical Bayesian math, identical prior — the
  *only* difference is rules vs. LLM reasoning and active investigation.
- **Why removals, not additions:** we made APA *more* agentic by deleting tools that were
  RPA logic in disguise or redundant re-reads, not by adding more tools.
- **Coverage, not count:** 7 tools chosen because every failure category has a decisive
  channel — not because 7 is a target.

---

## Pending (not yet done at time of writing)

- Full eval re-run — every change above (taxonomy=9, new prior, removed tools, K_max=5,
  framed tools, real web/log-diff updates) shifts both cost and accuracy.
- Thesis tool-library + prompt-design sections rewritten to the 7-tool design.
- Architecture diagram: stopping diamond `step ≥ 7` → `step ≥ 5`; tool battery 7 tools.
- Optional: add `signal_previous_runs` to RPA battery for exact evidence parity (RPA would
  become 10 signals), or note the asymmetry in text.
- Future work hook: a source-patch inspection tool (read changed-function hunks, link to
  failing test names) would extend the same evidence-channel pattern.
