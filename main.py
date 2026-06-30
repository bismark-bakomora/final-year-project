"""
Heart Disease Prediction — CLI entry point.

Paper-aligned, resource-safe pipeline. Run one experiment at a time;
artifacts and logs are persisted under outputs/.

Examples:
  python main.py preprocess
  python main.py run hybrid --evaluate            # ~35 min (paper preset default)
  python main.py run compare --evaluate           # Table 7, staged (~2.5 h)
  python main.py run standalone gwo --evaluate
  python main.py run compare --preset quick --evaluate
  python main.py evaluate --run-id latest
  python main.py shap --run-id latest
  python main.py smote --run-id latest
"""

from __future__ import annotations

import argparse
import sys
import io

# UTF-8 console on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

# Disable oneDNN before TensorFlow loads — prevents MKL OOM on long CPU runs
from src.tf_config import configure_tensorflow_env

configure_tensorflow_env()


def _import_pipeline():
    from src.pipeline import PipelineRunner, load_runner_for_eval
    return PipelineRunner, load_runner_for_eval


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GWO-WOA-AOA heart disease prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Shared flags — available after the subcommand (e.g. run compare --preset paper)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--preset",
        choices=["paper", "quick"],
        default="paper",
        help="paper = Section 4.2 settings; quick = smoke test (default: paper)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser(
        "preprocess",
        parents=[shared],
        help="Prepare dataset (Section 3.1)",
    )
    preprocess.add_argument(
        "--run-id",
        default=None,
        help="Custom run ID (default: UTC timestamp)",
    )

    run = sub.add_parser(
        "run",
        parents=[shared],
        help="Train models (one mode per invocation)",
    )
    run.add_argument(
        "--run-id",
        default=None,
        help="Custom run ID (default: UTC timestamp)",
    )
    run.add_argument(
        "mode",
        choices=["hybrid", "baseline", "standalone", "compare"],
        help=(
            "hybrid = GWO-WOA-AOA (~35 min); "
            "compare = all Table 7 models sequentially; "
            "standalone = one algorithm; "
            "baseline = NO-CNN only"
        ),
    )
    run.add_argument(
        "--algorithm",
        choices=["gwo", "woa", "aoa", "rime"],
        help="Required when mode=standalone",
    )
    run.add_argument(
        "--evaluate",
        action="store_true",
        help="Run test-set evaluation after training",
    )
    run.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Re-run preprocessing even if data exists",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="Resume compare run: skip models already saved in --run-id",
    )

    ev = sub.add_parser(
        "evaluate",
        parents=[shared],
        help="Evaluate saved models from a previous run",
    )
    ev.add_argument(
        "--run-id",
        default="latest",
        help="Run ID or 'latest' (default)",
    )

    shap = sub.add_parser(
        "shap",
        parents=[shared],
        help="SHAP explainability on hybrid model (Section 4.4)",
    )
    shap.add_argument("--run-id", default="latest")

    smote = sub.add_parser(
        "smote",
        parents=[shared],
        help="SMOTE augmentation study (Section 4.5)",
    )
    smote.add_argument("--run-id", default="latest")

    return parser


def cmd_preprocess(args: argparse.Namespace) -> int:
    PipelineRunner, _ = _import_pipeline()
    runner = PipelineRunner.create(
        preset=args.preset, run_id=args.run_id, mode="preprocess"
    )
    runner.ensure_preprocessed(force=True)
    runner.finish()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    PipelineRunner, _ = _import_pipeline()
    if args.mode == "standalone" and not args.algorithm:
        print("error: --algorithm is required for mode=standalone", file=sys.stderr)
        return 2

    runner = PipelineRunner.create(
        preset=args.preset,
        run_id=args.run_id,
        mode=args.mode,
        resume=args.resume,
    )
    try:
        runner.ensure_preprocessed(force=args.force_preprocess)
        data = runner.load_data()

        if args.mode == "hybrid":
            runner.run_hybrid(data)
        elif args.mode == "baseline":
            runner.run_baseline(data)
        elif args.mode == "standalone":
            runner.run_standalone(data, args.algorithm)
        elif args.mode == "compare":
            runner.run_compare(data, resume=args.resume)

        if args.evaluate or args.mode in ("hybrid", "compare"):
            runner.evaluate_run(data)

        runner.finish()
        print(f"\nRun complete. Artifacts: {runner.run_dir}")
        print(f"Log file: {runner.settings.logs_dir / (runner.run_id + '.log')}")
        return 0
    except Exception:
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    _, load_runner_for_eval = _import_pipeline()
    runner, _ = load_runner_for_eval(args.preset, args.run_id)
    data = runner.load_data()
    runner.evaluate_run(data)
    runner.finish()
    return 0


def cmd_shap(args: argparse.Namespace) -> int:
    _, load_runner_for_eval = _import_pipeline()
    runner, _ = load_runner_for_eval(args.preset, args.run_id)
    data = runner.load_data()
    runner.run_shap(data)
    runner.finish()
    return 0


def cmd_smote(args: argparse.Namespace) -> int:
    _, load_runner_for_eval = _import_pipeline()
    runner, _ = load_runner_for_eval(args.preset, args.run_id)
    data = runner.load_data()
    runner.run_smote(data)
    runner.finish()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "preprocess": cmd_preprocess,
        "run": cmd_run,
        "evaluate": cmd_evaluate,
        "shap": cmd_shap,
        "smote": cmd_smote,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
