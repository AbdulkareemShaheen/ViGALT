#!/usr/bin/env python3
"""
Run the full ViGALT workflow from an Amazon product URL.

Steps: extract DOM + image → 7-stage pipeline → export claims → evaluation.

Usage:
  python -m atsn.run_from_url --url "https://www.amazon.fr/dp/B077XM3DV5"
  python -m atsn.run_from_url --url "..." --run-id 5
  python -m atsn.run_from_url --url "..." --work-dir output/runs/3
  python -m atsn.run_from_url --html saved_page.html --work-dir output/runs/3
  python -m atsn.run_from_url --url "..." --skip-eval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ..evaluation.utils import export_claims_from_pipeline
from ..paths import project_root, resolve_path
from ..run_layout import RunPaths, allocate_run_dir


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViGALT end-to-end from an Amazon product URL."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Amazon product page URL.")
    source.add_argument("--html", help="Locally saved Amazon product HTML page.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for all run outputs (overrides --run-id).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Use output/runs/N as the run folder (default: next free number).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Stop after pipeline + claim export (skip evaluation metrics).",
    )
    parser.add_argument(
        "--single-model",
        default=None,
        metavar="MODEL",
        help="Override all OpenAI stage models with a single model.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-stage OpenAI timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between pipeline/evaluator requests (default: 3.0).",
    )
    parser.add_argument(
        "--extract-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout for Amazon DOM extraction (default: 30).",
    )
    return parser.parse_args()


def resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    if args.work_dir:
        return allocate_run_dir(work_dir=resolve_path(args.work_dir))
    return allocate_run_dir(run_id=args.run_id)


def run_command(label: str, cmd: list[str]) -> int:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=project_root())
    if result.returncode != 0:
        print(f"Error: step failed ({label})", file=sys.stderr)
    return result.returncode


def export_claims_step(paths: RunPaths) -> int:
    print("\n" + "=" * 60)
    print("Step 3/7 — Export claims for evaluation")
    print("=" * 60)
    try:
        export_claims_from_pipeline(
            paths.pipeline_json,
            paths.claims_json,
            product_stem=paths.run_id,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def metric_from_results(path: Path, *keys: str) -> object | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    results = record.get("results") or []
    if not results:
        return None
    evaluation = results[0].get("evaluation") or {}
    summary = evaluation.get("summary") or evaluation
    for key in keys:
        if key in summary:
            return summary.get(key)
    return None


def print_summary(paths: RunPaths) -> None:
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Run folder:     {paths.work_dir}")

    if paths.pipeline_json.exists():
        pipeline = json.loads(paths.pipeline_json.read_text(encoding="utf-8"))
        final_alt = pipeline.get("final_alt_text") or "(none)"
        print(f"Final alt text: {final_alt}")
        print(f"Pipeline JSON:  {paths.pipeline_json}")

    image_path = paths.resolve_image_path()
    artifacts = [
        ("DOM", paths.dom_json),
        *([("Image", image_path)] if image_path else []),
        ("Meta", paths.meta_json),
        ("Claims", paths.claims_json),
        ("Relevancy", paths.relevancy_json),
        ("Redundancy", paths.redundancy_json),
        ("Objectivity", paths.objectivity_json),
        ("Efficiency", paths.efficiency_json),
    ]
    for label, path in artifacts:
        if path.exists():
            print(f"{label + ':':14} {path}")

    if paths.relevancy_json.exists():
        print(f"Relevancy:      {metric_from_results(paths.relevancy_json, 'relevancy_score')}")
    if paths.objectivity_json.exists():
        print(f"Objectivity:    {metric_from_results(paths.objectivity_json, 'objectivity_score')}")
    if paths.redundancy_json.exists():
        print(f"Novelty rate:   {metric_from_results(paths.redundancy_json, 'novelty_rate')}")
    if paths.efficiency_json.exists():
        print(f"Efficiency:     {metric_from_results(paths.efficiency_json, 'efficiency_per_100_words')}")


def run(args: argparse.Namespace) -> int:
    configure_stdout()

    if not args.skip_eval and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is required unless --skip-eval is set.", file=sys.stderr)
        return 1

    paths = resolve_run_paths(args)
    paths.work_dir.mkdir(parents=True, exist_ok=True)

    extract_cmd = [
        sys.executable,
        "-m",
        "atsn.cli.extract_dom",
        "--output-dir",
        str(paths.work_dir),
        "--run-id",
        paths.run_id,
        "--timeout",
        str(args.extract_timeout),
    ]
    if args.url:
        extract_cmd.extend(["--url", args.url])
    else:
        extract_cmd.extend(["--html", str(resolve_path(args.html))])

    if run_command("Step 1/7 — Extract DOM + image", extract_cmd) != 0:
        return 1
    if not paths.dom_json.exists():
        print(f"Error: expected product DOM at {paths.dom_json}", file=sys.stderr)
        return 1

    pipeline_cmd = [
        sys.executable,
        "-m",
        "atsn.pipeline",
        "--product",
        str(paths.dom_json),
        "--output",
        str(paths.pipeline_json),
        "--timeout",
        str(args.timeout),
        "--delay",
        str(args.delay),
    ]
    if args.single_model:
        pipeline_cmd.extend(["--single-model", args.single_model])

    if run_command("Step 2/7 — Generate alt text (7-stage pipeline)", pipeline_cmd) != 0:
        return 1

    if export_claims_step(paths) != 0:
        return 1

    if args.skip_eval:
        print_summary(paths)
        return 0

    eval_base = [
        "--algorithm",
        "our",
        "--products-dir",
        str(paths.work_dir),
        "--timeout",
        str(args.timeout),
        "--delay",
        str(args.delay),
    ]
    if args.single_model:
        eval_base.extend(["--model", args.single_model])

    eval_with_image = [
        *eval_base,
        "--images-dir",
        str(paths.work_dir),
    ]

    relevancy_cmd = [
        sys.executable,
        "-m",
        "atsn.evaluation.relevancy",
        *eval_with_image,
        "--our-claims-dir",
        str(paths.work_dir),
        "--output",
        str(paths.relevancy_json),
    ]
    if run_command("Step 4/7 — Relevancy evaluation", relevancy_cmd) != 0:
        return 1

    redundancy_cmd = [
        sys.executable,
        "-m",
        "atsn.evaluation.redundancy",
        *eval_base,
        "--relevancy-input",
        str(paths.relevancy_json),
        "--output",
        str(paths.redundancy_json),
    ]
    if run_command("Step 5/7 — Redundancy evaluation", redundancy_cmd) != 0:
        return 1

    objectivity_cmd = [
        sys.executable,
        "-m",
        "atsn.evaluation.objectivity",
        *eval_with_image,
        "--relevancy-input",
        str(paths.relevancy_json),
        "--output",
        str(paths.objectivity_json),
    ]
    if run_command("Step 6/7 — Objectivity evaluation", objectivity_cmd) != 0:
        return 1

    efficiency_cmd = [
        sys.executable,
        "-m",
        "atsn.evaluation.efficiency",
        "--algorithm",
        "our",
        "--redundancy-input",
        str(paths.redundancy_json),
        "--relevancy-input",
        str(paths.relevancy_json),
        "--output",
        str(paths.efficiency_json),
    ]
    if run_command("Step 7/7 — Efficiency evaluation", efficiency_cmd) != 0:
        return 1

    print_summary(paths)
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
