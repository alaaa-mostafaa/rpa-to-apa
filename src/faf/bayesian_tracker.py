import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from faf.models import RunEvent

# ─── failure categories ─────────

CATEGORIES = [
    "CODE_REGRESSION",
    "DEPENDENCY_CONFLICT",
    "CONFIG_ERROR",
    "ENV_FLAKINESS",
    "TEST_FLAKINESS",
    "TOOLING_ARTIFACT",
    "CASCADE_FAILURE",
    "INFRA_INCOMPATIBILITY",
    "NETWORK_TRANSIENT",
    "OOM_KILL",
    "POD_CRASH",
    "IMAGE_PULL_BACKOFF",
    "SCHEDULING_ERROR",
    "UNKNOWN",
]

N_CATEGORIES = len(CATEGORIES)

# ─── Signal Registry ─────────────────────────────────────────────

class SignalRegistry:
    """Registry for diagnostic signals that evaluate a RunEvent into likelihoods."""
    def __init__(self):
        self._signals: Dict[str, Callable[[RunEvent], Dict[str, float]]] = {}

    def register(self, name: str):
        def decorator(func: Callable[[RunEvent], Dict[str, float]]):
            self._signals[name] = func
            return func
        return decorator

    def evaluate(self, event: RunEvent, signal_name: str) -> Dict[str, float]:
        """Evaluates a specific signal against a RunEvent."""
        if signal_name in self._signals:
            return self._signals[signal_name](event)
        # Uniform fallback if signal is missing or unsupported
        return {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}
        
    def evaluate_available(self, event: RunEvent) -> Dict[str, Dict[str, float]]:
        """Evaluates all signals available on the RunEvent."""
        results = {}
        for sig in event.available_signals:
            if sig in self._signals:
                results[sig] = self._signals[sig](event)
        return results

# The global registry instance
registry = SignalRegistry()

# ─── signal definitions ─────────────────────────────────────────────

@registry.register("jobs_failed")
def signal_many_jobs_failed(event: RunEvent) -> Dict[str, float]:
    """Signal: how many jobs failed out of total?"""
    n_failed = event.metadata.get("jobs_failed", 0)
    n_total = event.metadata.get("jobs_total", 0)
    
    base = {cat: 0.10 for cat in CATEGORIES}
    ratio = n_failed / max(n_total, 1)

    if ratio > 0.8 and n_total >= 4:
        base["CASCADE_FAILURE"] += 0.05
        base["INFRA_INCOMPATIBILITY"] += 0.03
        base["ENV_FLAKINESS"] += 0.02
    elif ratio < 0.3 and n_total >= 3:
        base["CODE_REGRESSION"] += 0.04
        base["TEST_FLAKINESS"] += 0.03

    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


@registry.register("error_text")
def signal_error_text(event: RunEvent) -> Dict[str, float]:
    """Signal: what patterns appear in the error text?"""
    error_texts = [step.error_text.lower() for step in event.failed_steps if step.error_text]
    if not error_texts:
        return {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}

    base = {cat: 0.05 for cat in CATEGORIES}
    text = " ".join(error_texts)

    # Infrastructure / compatibility
    if any(w in text for w in ("glibc", "libc.so", "libstdc", "libm.so")):
        base["INFRA_INCOMPATIBILITY"] += 0.35
    if any(w in text for w in ("node20", "node18", "node16", "node12")):
        base["INFRA_INCOMPATIBILITY"] += 0.20

    # Network / transient
    if any(w in text for w in ("connection reset", "connection refused", "timeout", "timed out", 
                               "etimedout", "econnreset", "econnrefused", "socket hang up", 
                               "network error", "could not resolve host", "ssl_error", "certificate")):
        base["NETWORK_TRANSIENT"] += 0.30
        base["ENV_FLAKINESS"] += 0.10

    # Dependency issues
    if any(w in text for w in ("no module named", "modulenotfounderror", "importerror", "cannot find module",
                               "could not resolve dependencies", "peer dependency", "npm err",
                               "pip install failed", "cargo error", "unresolved dependency",
                               "version conflict", "incompatible", "requires python", "requires java")):
        base["DEPENDENCY_CONFLICT"] += 0.25

    # Test failures
    if any(w in text for w in ("assert", "assertion", "expect", "test failed", "tests failed",
                               "pytest", "junit", "rspec", "mocha", "jest", "test result")):
        base["CODE_REGRESSION"] += 0.10
        base["TEST_FLAKINESS"] += 0.12
    elif any(w in text for w in ("fail ", "failed ", "failures")):
        base["CODE_REGRESSION"] += 0.03
        base["DEPENDENCY_CONFLICT"] += 0.03
        base["CONFIG_ERROR"] += 0.03

    # Build / compile errors
    if any(w in text for w in ("syntax error", "syntaxerror", "compile error", "compilation failed",
                               "undefined reference", "linker error", "unexpected token", "parse error")):
        base["CODE_REGRESSION"] += 0.25
    if any(w in text for w in ("build failed", "build failure")):
        base["CODE_REGRESSION"] += 0.08
        base["DEPENDENCY_CONFLICT"] += 0.06
        base["CONFIG_ERROR"] += 0.04

    # SDK/project-structure conflicts
    if any(w in text for w in ("netsdk", "found multiple publish output", "duplicate class", 
                               "duplicate file", "conflicting files", "multiple artifacts")):
        base["CODE_REGRESSION"] += 0.25
        base["DEPENDENCY_CONFLICT"] -= 0.05

    # Config errors
    if any(w in text for w in ("invalid workflow", "deprecated action", "missing input", 
                               "unexpected value", "input.*required", "workflow.*invalid")):
        base["CONFIG_ERROR"] += 0.20
    if any(w in text for w in ("permission denied", "not authorized", "env variable", "environment variable")):
        base["CONFIG_ERROR"] += 0.08
        base["CODE_REGRESSION"] += 0.04

    # Dependabot
    if any(w in text for w in ("dependabot", "read-only access", "dependabot on the", "workflows triggered by dependabot")):
        base["DEPENDENCY_CONFLICT"] += 0.30
        base["CONFIG_ERROR"] -= 0.10

    # Git errors
    if any(w in text for w in ("not a git repository", "fatal: not a git")):
        base["CODE_REGRESSION"] += 0.15
        base["CONFIG_ERROR"] -= 0.10

    # Cancellation
    if any(w in text for w in ("canceled", "cancelled", "operation was canceled")):
        base["CASCADE_FAILURE"] += 0.15

    # Tooling artifacts
    if any(w in text for w in ("bashword", "circular structure", "parser exception", "bash-command-extractor")):
        base["TOOLING_ARTIFACT"] += 0.40

    # Resource exhaustion
    if any(w in text for w in ("out of memory", "oom", "killed", "no space left", "disk full",
                               "resource exhausted", "quota exceeded")):
        base["ENV_FLAKINESS"] += 0.25
        base["OOM_KILL"] += 0.25

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}


@registry.register("branch_type")
def signal_branch_type(event: RunEvent) -> Dict[str, float]:
    """Signal: what kind of branch is this?"""
    branch = event.metadata.get("branch", "").lower()
    is_protected = event.metadata.get("is_protected_branch", False)
    
    base = {cat: 0.10 for cat in CATEGORIES}

    if any(bot in branch for bot in ("dependabot", "renovate")):
        base["DEPENDENCY_CONFLICT"] += 0.15
        base["INFRA_INCOMPATIBILITY"] += 0.05
    elif not is_protected and branch:
        base["CODE_REGRESSION"] += 0.03

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}


@registry.register("commit_message")
def signal_commit_message(event: RunEvent) -> Dict[str, float]:
    """Signal: what does the commit message suggest?"""
    msg = event.metadata.get("commit_message", "").lower()
    base = {cat: 0.09 for cat in CATEGORIES}
    
    if not msg:
        return {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}

    if any(w in msg for w in ("upgrade", "bump", "update dep", "update version", "update dependency")):
        base["DEPENDENCY_CONFLICT"] += 0.15
        base["INFRA_INCOMPATIBILITY"] += 0.05
    if any(w in msg for w in ("fix ", "fix:", "fixed ", "hotfix", "patch ", "repair ", "resolve ", "bug ", "bugfix")):
        base["CODE_REGRESSION"] += 0.05
        base["CONFIG_ERROR"] += 0.03
        base["DEPENDENCY_CONFLICT"] += 0.03
    if any(w in msg for w in ("refactor", "rename", "move ", "reorganize", "cleanup", "clean up")):
        base["CODE_REGRESSION"] += 0.04
        base["CONFIG_ERROR"] += 0.04
    if any(w in msg for w in ("ci ", "ci:", "workflow", "action", "yaml", "yml", "pipeline", "github action")):
        base["CONFIG_ERROR"] += 0.15
    if any(w in msg for w in ("test ", "test:", "spec ", "coverage", "add test", "fix test")):
        base["TEST_FLAKINESS"] += 0.08
        base["CODE_REGRESSION"] += 0.05
    if any(w in msg for w in ("docker", "container", "image ", "dockerfile")):
        base["INFRA_INCOMPATIBILITY"] += 0.10
        base["CONFIG_ERROR"] += 0.05
    if any(w in msg for w in ("merge ", "merge:", "revert ")):
        base["CODE_REGRESSION"] += 0.05
    if any(w in msg for w in ("wip", "tmp", "temp", "draft")):
        base["CODE_REGRESSION"] += 0.08

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}


@registry.register("previous_runs")
def signal_previous_runs(event: RunEvent) -> Dict[str, float]:
    """Signal: how often has this workflow been failing recently?"""
    n_recent_failures = event.metadata.get("recent_failures", 0)
    n_recent_total = event.metadata.get("recent_total", 0)
    
    base = {cat: 0.10 for cat in CATEGORIES}

    if n_recent_total == 0:
        return base

    failure_rate = n_recent_failures / n_recent_total
    if failure_rate > 0.5:
        base["ENV_FLAKINESS"] += 0.12
        base["TEST_FLAKINESS"] += 0.08
        base["CONFIG_ERROR"] += 0.05
    elif failure_rate > 0.2:
        base["TEST_FLAKINESS"] += 0.05
        base["ENV_FLAKINESS"] += 0.03
    else:
        base["CODE_REGRESSION"] += 0.08
        base["DEPENDENCY_CONFLICT"] += 0.05

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}


@registry.register("parent_commit_run")
def signal_parent_commit_run(event: RunEvent) -> Dict[str, float]:
    """Signal: what was the conclusion of the run on the immediate parent commit?"""
    def _normalise(raw: Dict[str, float]) -> Dict[str, float]:
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    parent_conclusion = event.metadata.get("parent_run_conclusion")

    if parent_conclusion == "success":
        return _normalise({
            "CODE_REGRESSION":       0.55,
            "DEPENDENCY_CONFLICT":   0.30,
            "CONFIG_ERROR":          0.20,
            "ENV_FLAKINESS":         0.03,
            "TEST_FLAKINESS":        0.03,
            "TOOLING_ARTIFACT":      0.06,
            "CASCADE_FAILURE":       0.06,
            "INFRA_INCOMPATIBILITY": 0.03,
            "NETWORK_TRANSIENT":     0.04,
            "UNKNOWN":               0.08,
            "OOM_KILL":              0.02,
            "POD_CRASH":             0.02,
            "IMAGE_PULL_BACKOFF":    0.02,
            "SCHEDULING_ERROR":      0.02,
        })
    elif parent_conclusion == "failure":
        return _normalise({
            "CODE_REGRESSION":       0.03,
            "DEPENDENCY_CONFLICT":   0.05,
            "CONFIG_ERROR":          0.06,
            "ENV_FLAKINESS":         0.30,
            "TEST_FLAKINESS":        0.40,
            "TOOLING_ARTIFACT":      0.15,
            "CASCADE_FAILURE":       0.15,
            "INFRA_INCOMPATIBILITY": 0.25,
            "NETWORK_TRANSIENT":     0.15,
            "UNKNOWN":               0.10,
            "OOM_KILL":              0.05,
            "POD_CRASH":             0.05,
            "IMAGE_PULL_BACKOFF":    0.05,
            "SCHEDULING_ERROR":      0.05,
        })

    return {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}


@registry.register("detection_mode")
def signal_detection_mode(event: RunEvent) -> Dict[str, float]:
    """Signal: how was the failure detected?"""
    mode = event.metadata.get("detection_mode", "unknown_failure")
    base = {cat: 0.10 for cat in CATEGORIES}

    if mode == "per_step_error":
        base["CODE_REGRESSION"] += 0.05
        base["DEPENDENCY_CONFLICT"] += 0.03
    elif mode == "single_step_inferred":
        base["CONFIG_ERROR"] += 0.05
    elif mode == "job_level_fallback":
        base["CASCADE_FAILURE"] += 0.03
        base["ENV_FLAKINESS"] += 0.02
    elif mode == "unknown_failure":
        base["UNKNOWN"] += 0.10

    total = sum(base.values())
    return {k: max(v / total, 0.001) for k, v in base.items()}

# ─── the tracker ─────────────────────────────────────────────────────

@dataclass
class BeliefState:
    """Current probability distribution over failure categories."""
    probabilities: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.probabilities:
            self.probabilities = {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}

    def entropy(self) -> float:
        return -sum(p * math.log2(p) for p in self.probabilities.values() if p > 0)

    def max_entropy(self) -> float:
        return math.log2(N_CATEGORIES)

    def confidence(self) -> float:
        return 1.0 - (self.entropy() / self.max_entropy())

    def top_category(self) -> Tuple[str, float]:
        best = max(self.probabilities, key=self.probabilities.get)
        return best, self.probabilities[best]

    def top_n(self, n: int = 3) -> List[Tuple[str, float]]:
        sorted_cats = sorted(self.probabilities.items(), key=lambda x: -x[1])
        return sorted_cats[:n]

    def update(self, likelihood: Dict[str, float], signal_name: str = "") -> None:
        new_probs = {}
        for cat in CATEGORIES:
            new_probs[cat] = self.probabilities[cat] * likelihood.get(cat, 0.1)

        total = sum(new_probs.values())
        if total > 0:
            new_probs = {k: v / total for k, v in new_probs.items()}
        else:
            new_probs = {cat: 1.0 / N_CATEGORIES for cat in CATEGORIES}

        old_entropy = self.entropy()
        self.probabilities = new_probs
        new_entropy = self.entropy()

        self.history.append({
            "signal": signal_name,
            "top_3": self.top_n(3),
            "entropy": new_entropy,
            "information_gain": old_entropy - new_entropy,
            "confidence": self.confidence(),
        })

    def expected_information_gain(self, possible_likelihoods: List[Dict[str, float]]) -> float:
        current_entropy = self.entropy()
        expected_posterior_entropy = 0.0

        for likelihood in possible_likelihoods:
            new_probs = {}
            for cat in CATEGORIES:
                new_probs[cat] = self.probabilities[cat] * likelihood.get(cat, 0.1)
            total = sum(new_probs.values())
            if total > 0:
                new_probs = {k: v / total for k, v in new_probs.items()}
                entropy = -sum(p * math.log2(p) for p in new_probs.values() if p > 0)
            else:
                entropy = current_entropy
            expected_posterior_entropy += entropy

        if possible_likelihoods:
            expected_posterior_entropy /= len(possible_likelihoods)
        return current_entropy - expected_posterior_entropy
