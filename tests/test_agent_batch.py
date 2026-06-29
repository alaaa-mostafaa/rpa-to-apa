# test_agent_batch.py
# Run the LangGraph agent on a handful of diverse cases.

import gzip
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from agent import run_agent, print_agent_result

load_dotenv()

RUNS_PATH = Path("/home/guc_alaa/runs.json.gz")
OUTPUT_PATH = Path("/home/guc_alaa/test_agent_batch_results.json")

CASES = [
    # ── original 5 ───────────────────────────────────────────────────────
    {
        "label": "pyca/bcrypt — GLIBC incompatibility (Python wheels)",
        "run_id": "pyca/bcrypt_.github/workflows/wheel-builder.yml_82_1",
    },
    {
        "label": "apache/unomi — network blip during Maven download (Java)",
        "run_id": "apache/unomi_.github/workflows/unomi-ci-build-tests.yml_1369_1",
    },
    {
        "label": "ohdsi/dataqualitydashboard — R CMD check test failure",
        "run_id": "ohdsi/dataqualitydashboard_.github/workflows/R_CMD_check_main_weekly.yaml_28_1",
    },
    {
        "label": "hibernate/hibernate-search — Java ORM upgrade regression",
        "run_id": "hibernate/hibernate-search_.github/workflows/simple-build.yml_66_1",
    },
    {
        "label": "devfile/api — outdated GitHub Actions / Go syntax error",
        "run_id": "devfile/api_.github/workflows/codecov.yaml_23_1",
    },
    # ── 15 new diverse cases ─────────────────────────────────────────────
    # test
    {
        "label": "rtkconsortium/rtk — Python CUDA wheel build+test",
        "run_id": "rtkconsortium/rtk_.github/workflows/build-test-package-python-cuda.yml_133_1",
    },
    {
        "label": "arelle/arelle — Python UI test runner failure",
        "run_id": "arelle/arelle_.github/workflows/test-ui.yml_487_1",
    },
    {
        "label": "ndd7xv/heh — Rust check after clap dep bump",
        "run_id": "ndd7xv/heh_.github/workflows/check.yml_109_1",
    },
    {
        "label": "niklasei/bevy_game_template — iOS TestFlight release failure",
        "run_id": "niklasei/bevy_game_template_.github/workflows/release-ios-testflight.yaml_18_1",
    },
    # build
    {
        "label": "ttauri-project/ttauri — C++ Windows MSVC build failure",
        "run_id": "ttauri-project/ttauri_.github/workflows/build-on-windows.yml_2924_1",
    },
    {
        "label": "unblockneteasemusic/server-rust — Rust REST API build",
        "run_id": "unblockneteasemusic/server-rust_.github/workflows/rest-api-build.yml_387_1",
    },
    {
        "label": "dynamorio/dynamorio — CI package build (C/C++ cross-arch)",
        "run_id": "dynamorio/dynamorio_.github/workflows/ci-package.yml_175_1",
    },
    {
        "label": "admin-shell-io/aasx-package-explorer — .NET release build",
        "run_id": "admin-shell-io/aasx-package-explorer_.github/workflows/build-and-package-release.yml_46_1",
    },
    # CI / integration
    {
        "label": "dotcms/core — Java Maven CI/CD pipeline failure",
        "run_id": "dotcms/core_.github/workflows/maven-cicd-pipeline.yml_2954_1",
    },
    {
        "label": "tony133/nestjs-apps-collection — Node.js NX cloud CI",
        "run_id": "tony133/nestjs-apps-collection_.github/workflows/nx-cloud-ci.yml_413_1",
    },
    # lint / style
    {
        "label": "cubefs/chubaofs — Go blobstore format/lint check",
        "run_id": "cubefs/chubaofs_.github/workflows/blobstore_format.yml_392_1",
    },
    {
        "label": "apipie/apipie-rails — Ruby RuboCop challenger lint",
        "run_id": "apipie/apipie-rails_.github/workflows/rubocop-challenger.yml_91_1",
    },
    # deploy / release
    {
        "label": "securego/gosec — Go release workflow failure",
        "run_id": "securego/gosec_.github/workflows/release.yml_24_1",
    },
    {
        "label": "lukemathwalker/cargo-chef — Docker build (Rust, actions bump)",
        "run_id": "lukemathwalker/cargo-chef_.github/workflows/docker.yml_1128_1",
    },
    # scheduled / security / other
    {
        "label": "mlrun/mlrun — scheduled security scan failure",
        "run_id": "mlrun/mlrun_.github/workflows/security_scan.yaml_16_1",
    },
    {
        "label": "obsproject/obs-studio — CI dispatch workflow failure",
        "run_id": "obsproject/obs-studio_.github/workflows/dispatch.yaml_5_1",
    },
    {
        "label": "project-zot/zot — Go CI/CD (Trivy CVE scan integration)",
        "run_id": "project-zot/zot_.github/workflows/ci-cd.yml_8403_1",
    },
    {
        "label": "crate-ci/cargo-release — Rust scheduled rust-next nightly check",
        "run_id": "crate-ci/cargo-release_.github/workflows/rust-next.yml_27_1",
    },
]


def _build_summary(results: list) -> dict:
    total = len(results)
    fix_exact = fix_unclear = fix_none = 0
    cases_with_files = 0
    action_dist: dict = {}
    category_dist: dict = {}
    confidences = []
    bayes_confidences = []
    steps_list = []
    tool_freq: dict = {}

    for r in results:
        cl = r["result"]["classification"]

        fs = cl.get("fix_suggestion") or {}
        ffile = fs.get("file", "(none)") if isinstance(fs, dict) else "(none)"
        if ffile and ffile not in ("(unclear)", "(none)"):
            fix_exact += 1
        elif ffile == "(unclear)":
            fix_unclear += 1
        else:
            fix_none += 1

        mf = cl.get("mentioned_files") or r["result"].get("mentioned_files") or []
        if mf:
            cases_with_files += 1

        action = cl.get("action", "UNKNOWN")
        action_dist[action] = action_dist.get(action, 0) + 1

        cat = cl.get("category", "UNKNOWN")
        category_dist[cat] = category_dist.get(cat, 0) + 1

        conf = cl.get("confidence")
        if isinstance(conf, (int, float)):
            confidences.append(conf)

        bconf = cl.get("bayesian_confidence")
        if isinstance(bconf, (int, float)):
            bayes_confidences.append(bconf)

        steps = cl.get("steps_taken")
        if isinstance(steps, (int, float)):
            steps_list.append(steps)

        for tool in cl.get("tools_used") or []:
            tool_freq[tool] = tool_freq.get(tool, 0) + 1

    cases_with_changed_files = sum(
        1 for r in results
        if r["result"].get("changed_files")
    )

    return {
        "total_cases": total,
        "fix_suggestion": {
            "exact_file": fix_exact,
            "unclear": fix_unclear,
            "none": fix_none,
        },
        "cases_with_mentioned_files": cases_with_files,
        "cases_with_changed_files": cases_with_changed_files,
        "action_distribution": dict(sorted(action_dist.items())),
        "category_distribution": dict(sorted(category_dist.items())),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "avg_bayesian_confidence": sum(bayes_confidences) / len(bayes_confidences) if bayes_confidences else 0.0,
        "avg_steps_taken": sum(steps_list) / len(steps_list) if steps_list else 0.0,
        "tool_usage_frequency": dict(sorted(tool_freq.items(), key=lambda x: -x[1])),
    }


def main():
    print(f"Finding {len(CASES)} runs...")
    wanted = {c["run_id"] for c in CASES}
    raw_runs = {}
    with gzip.open(RUNS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = run.get("_id")
            if rid in wanted:
                raw_runs[rid] = run
                if len(raw_runs) == len(wanted):
                    break
    print(f"Found {len(raw_runs)}/{len(CASES)} runs.\n")

    results = []
    for case in CASES:
        print("█" * 70)
        print(f"  {case['label']}")
        print("█" * 70)

        raw = raw_runs.get(case["run_id"])
        if not raw:
            print("  NOT FOUND\n")
            continue

        try:
            result = run_agent(raw)
            print_agent_result(result)
            results.append({
                "label": case["label"],
                "result": result,
            })
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        cl = r["result"]["classification"]
        print(f"\n  {r['label']}")
        print(f"    {cl.get('category')} → {cl.get('action')}")
        print(f"    confidence: {cl.get('confidence', 0):.0%} "
              f"(bayes: {cl.get('bayesian_top')} @ {cl.get('bayesian_confidence', 0):.0%})")
        print(f"    steps: {cl.get('steps_taken')}, "
              f"tools: {', '.join(cl.get('tools_used', []))}")

    summary = _build_summary(results)

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"  Total cases : {summary['total_cases']}")
    print(f"  Fix suggestion breakdown:")
    fs = summary["fix_suggestion"]
    print(f"    exact file : {fs['exact_file']}")
    print(f"    unclear    : {fs['unclear']}")
    print(f"    none       : {fs['none']}")
    print(f"  Cases with mentioned_files : {summary['cases_with_mentioned_files']}")
    print(f"  Cases with changed_files   : {summary['cases_with_changed_files']}")
    print(f"  Action distribution : {summary['action_distribution']}")
    print(f"  Category distribution : {summary['category_distribution']}")
    print(f"  Avg confidence : {summary['avg_confidence']:.0%}")
    print(f"  Avg Bayesian confidence : {summary['avg_bayesian_confidence']:.0%}")
    print(f"  Avg steps taken : {summary['avg_steps_taken']:.1f}")
    print(f"  Tool usage : {summary['tool_usage_frequency']}")

    output = {"summary": summary, "cases": results}
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved detailed results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
