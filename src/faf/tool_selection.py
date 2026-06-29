from faf.bayesian_tracker import CATEGORIES, BeliefState

def _scenario(*bumps: tuple[str, float]) -> dict[str, float]:
    """Build a normalised likelihood vector from category bumps."""
    base = {cat: 0.05 for cat in CATEGORIES}
    for cat, weight in bumps:
        if cat in base:
            base[cat] += weight
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}

OUTCOME_SCENARIOS: dict[str, list[dict[str, float]]] = {
    "deep_log_analysis": [
        _scenario(("CODE_REGRESSION",       0.55), ("TEST_FLAKINESS",      0.10)),
        _scenario(("DEPENDENCY_CONFLICT",   0.55), ("CONFIG_ERROR",        0.10)),
        _scenario(("INFRA_INCOMPATIBILITY", 0.50), ("ENV_FLAKINESS",       0.10)),
        _scenario(("TEST_FLAKINESS",        0.45), ("ENV_FLAKINESS",       0.15)),
        _scenario(("NETWORK_TRANSIENT",     0.50), ("ENV_FLAKINESS",       0.15)),
        _scenario(),
    ],
    "inspect_commit_diff": [
        _scenario(("DEPENDENCY_CONFLICT",   0.45), ("CONFIG_ERROR",        0.10)),
        _scenario(("CONFIG_ERROR",          0.50), ("INFRA_INCOMPATIBILITY", 0.15)),
        _scenario(("CODE_REGRESSION",       0.40), ("TEST_FLAKINESS",      0.10)),
        _scenario(("INFRA_INCOMPATIBILITY", 0.40), ("CONFIG_ERROR",        0.10)),
        _scenario(("UNKNOWN",               0.15)),
    ],
    "check_run_history": [
        _scenario(("CODE_REGRESSION",       0.45), ("DEPENDENCY_CONFLICT", 0.20),
                  ("CONFIG_ERROR",          0.15)),
        _scenario(("TEST_FLAKINESS",        0.40), ("ENV_FLAKINESS",       0.25),
                  ("INFRA_INCOMPATIBILITY", 0.15)),
        _scenario(("CASCADE_FAILURE",       0.25), ("ENV_FLAKINESS",       0.15)),
        _scenario(),
    ],
    "inspect_workflow_file": [
        _scenario(("INFRA_INCOMPATIBILITY", 0.55), ("CONFIG_ERROR",        0.10)),
        _scenario(("CONFIG_ERROR",          0.45), ("INFRA_INCOMPATIBILITY", 0.15)),
        _scenario(("ENV_FLAKINESS",         0.35), ("INFRA_INCOMPATIBILITY", 0.20)),
        _scenario(("UNKNOWN",               0.10)),
    ],
    "inspect_dependency_changes": [
        _scenario(("DEPENDENCY_CONFLICT",   0.65), ("CONFIG_ERROR",        0.10)),
        _scenario(("DEPENDENCY_CONFLICT",   0.55), ("CODE_REGRESSION",      0.10)),
        _scenario(("INFRA_INCOMPATIBILITY", 0.45), ("DEPENDENCY_CONFLICT",  0.25)),
        _scenario(),
    ],
    "inspect_runner_environment": [
        _scenario(("INFRA_INCOMPATIBILITY", 0.65), ("CONFIG_ERROR",         0.10)),
        _scenario(("DEPENDENCY_CONFLICT",   0.45), ("INFRA_INCOMPATIBILITY", 0.25)),
        _scenario(("ENV_FLAKINESS",         0.30), ("INFRA_INCOMPATIBILITY", 0.20)),
        _scenario(),
    ],
    "inspect_k8s_events": [
        _scenario(("OOM_KILL", 0.60), ("CONFIG_ERROR", 0.15)),
        _scenario(("IMAGE_PULL_BACKOFF", 0.65), ("CONFIG_ERROR", 0.15)),
        _scenario(("SCHEDULING_ERROR", 0.55), ("INFRA_INCOMPATIBILITY", 0.20)),
        _scenario(("POD_CRASH", 0.50), ("CODE_REGRESSION", 0.25)),
        _scenario(("UNKNOWN", 0.10)),
    ],
}

def compute_tool_eig(bs: BeliefState, tool: str) -> float:
    scenarios = OUTCOME_SCENARIOS.get(tool)
    if not scenarios:
        return 0.0
    return max(0.0, bs.expected_information_gain(scenarios))

def rank_tools_by_eig(bs: BeliefState, available_tools: list[str]) -> list[tuple[str, float]]:
    ranked = [(tool, compute_tool_eig(bs, tool)) for tool in available_tools]
    ranked.sort(key=lambda x: -x[1])

    top_cat = bs.top_category()[0]
    if top_cat in ("ENV_FLAKINESS", "TEST_FLAKINESS"):
        ranked = [
            (tool, eig * 2.5 if tool == "check_run_history" else eig)
            for tool, eig in ranked
        ]
        ranked.sort(key=lambda x: -x[1])

    return ranked

def format_eig_for_prompt(rankings: list[tuple[str, float]]) -> str:
    if not rankings:
        return "(no tools available)"
    lines = []
    for i, (tool, eig) in enumerate(rankings):
        annotation = "  <- highest expected gain" if i == 0 else ""
        if i == len(rankings) - 1 and len(rankings) > 1:
            annotation = "  <- lowest expected gain"
        lines.append(f"  {tool:<26} {eig:.3f} bits{annotation}")
    return "\\n".join(lines)

def pick_eig_tool(bs: BeliefState, available_tools: list[str]) -> tuple[str, float, list[tuple[str, float]]]:
    if not available_tools:
        return "classify", 0.0, []
    rankings = rank_tools_by_eig(bs, available_tools)
    best_tool, best_eig = rankings[0]
    return best_tool, best_eig, rankings
