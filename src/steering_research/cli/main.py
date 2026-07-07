from __future__ import annotations

import argparse
from pathlib import Path

from steering_research.cli.experiments import (
    run_e001,
    run_e002,
    run_e003,
    run_e004,
    run_e005,
    run_qwen_limited_smoke,
    run_real_smoke_suite,
    write_smoke_summary,
)
from steering_research.data import BenchmarkStore
from steering_research.reports.dashboard import write_static_dashboard
from steering_research.reports.verify import verify_runs
from steering_research.runtime.config import repo_root_from_cwd
from steering_research.training import run_lora_sft


def _validate(repo_root: Path) -> int:
    store = BenchmarkStore.from_repo_root(repo_root)
    result = store.validate()
    print(result)
    if result["missing_positive_refs"] or result["missing_negative_refs"]:
        return 1
    return 0


def _smoke(repo_root: Path, backend: str, limit: int | None) -> int:
    if backend == "qwen" and limit is not None:
        run_dir = run_qwen_limited_smoke(repo_root, limit)
        dashboard = write_static_dashboard(repo_root / "runs")
        print(f"qwen limited smoke passed; dashboard={dashboard}")
        print(run_dir)
        return 0
    run_dirs = [
        run_e001(
            repo_root, repo_root / "configs" / "experiments" / "e001_mean_direction.yaml", backend
        ),
        run_e002(
            repo_root,
            repo_root / "configs" / "experiments" / "e002_activation_monitor.yaml",
            backend,
        ),
        run_e003(repo_root, repo_root / "configs" / "experiments" / "e003_sae_delta.yaml", backend),
        run_e004(
            repo_root, repo_root / "configs" / "experiments" / "e004_steering_eval.yaml", backend
        ),
        run_e005(
            repo_root,
            repo_root / "configs" / "experiments" / "e005_sae_feature_steering.yaml",
            backend,
        ),
    ]
    write_smoke_summary(repo_root, run_dirs)
    dashboard = write_static_dashboard(repo_root / "runs")
    print(f"smoke passed; dashboard={dashboard}")
    for run_dir in run_dirs:
        print(run_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="steering")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-data")

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--backend", choices=["fake", "qwen"], default="fake")
    smoke.add_argument("--limit", type=int, default=None)

    smoke_real = sub.add_parser("smoke-real")
    smoke_real.add_argument("--examples", type=int, default=10)

    e001 = sub.add_parser("e001")
    e001.add_argument(
        "--config", type=Path, default=Path("configs/experiments/e001_mean_direction.yaml")
    )
    e001.add_argument("--backend", choices=["fake", "qwen"], default=None)

    e002 = sub.add_parser("e002")
    e002.add_argument(
        "--config", type=Path, default=Path("configs/experiments/e002_activation_monitor.yaml")
    )
    e002.add_argument("--backend", choices=["fake", "qwen"], default=None)

    e003 = sub.add_parser("e003")
    e003.add_argument(
        "--config", type=Path, default=Path("configs/experiments/e003_sae_delta.yaml")
    )
    e003.add_argument("--backend", choices=["fake", "qwen"], default=None)

    e004 = sub.add_parser("e004")
    e004.add_argument(
        "--config", type=Path, default=Path("configs/experiments/e004_steering_eval.yaml")
    )
    e004.add_argument("--backend", choices=["fake", "qwen"], default=None)

    e005 = sub.add_parser("e005")
    e005.add_argument(
        "--config", type=Path, default=Path("configs/experiments/e005_sae_feature_steering.yaml")
    )
    e005.add_argument("--backend", choices=["fake", "qwen"], default=None)

    e006 = sub.add_parser("e006-lora-sft")
    e006.add_argument("--config", type=Path, default=Path("configs/experiments/e006_lora_sft.yaml"))

    verify = sub.add_parser("verify-runs")
    verify.add_argument("--runs-root", type=Path, default=Path("runs"))

    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("--runs-root", type=Path, default=Path("runs"))

    args = parser.parse_args()
    repo_root = repo_root_from_cwd()
    if args.command == "validate-data":
        return _validate(repo_root)
    if args.command == "smoke":
        return _smoke(repo_root, str(args.backend), args.limit)
    if args.command == "smoke-real":
        if int(args.examples) != 100:
            print("smoke-real uses checked-in smoke configs; --examples is informational.")
        run_dirs = run_real_smoke_suite(repo_root)
        for run_dir in run_dirs:
            print(run_dir)
        return 0
    if args.command == "e001":
        print(run_e001(repo_root, repo_root / args.config, args.backend))
        return 0
    if args.command == "e002":
        print(run_e002(repo_root, repo_root / args.config, args.backend))
        return 0
    if args.command == "e003":
        print(run_e003(repo_root, repo_root / args.config, args.backend))
        return 0
    if args.command == "e004":
        print(run_e004(repo_root, repo_root / args.config, args.backend))
        return 0
    if args.command == "e005":
        print(run_e005(repo_root, repo_root / args.config, args.backend))
        return 0
    if args.command == "e006-lora-sft":
        print(run_lora_sft(repo_root, repo_root / args.config))
        return 0
    if args.command == "verify-runs":
        result = verify_runs(repo_root / args.runs_root)
        print(result)
        return 0 if result["ok"] else 1
    if args.command == "dashboard":
        print(write_static_dashboard(repo_root / args.runs_root))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
