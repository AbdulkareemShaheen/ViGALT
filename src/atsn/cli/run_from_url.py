#!/usr/bin/env python3
"""
Run the full ViGALT workflow from an Amazon product URL.

Steps: extract DOM + image → 7-stage pipeline → claim extraction → evaluation.

Usage:
  python -m atsn.run_from_url --url "https://www.amazon.fr/dp/B077XM3DV5"
  python -m atsn.run_from_url --url "..." --work-dir output/runs/B077XM3DV5
  python -m atsn.run_from_url --html saved_page.html --work-dir output/runs/B077XM3DV5
  python -m atsn.run_from_url --url "..." --skip-eval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ..extraction.amazon import extract_asin
from ..paths import project_root, resolve_path


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
        help="Working directory for DOM, image, and evaluation outputs.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Stop after pipeline + claim extraction (skip evaluation metrics).",
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


def resolve_work_dir(args: argparse.Namespace) -> tuple[Path, str]:
    if args.work_dir:
        work_dir = resolve_path(args.work_dir)
        asin = work_dir.name
    elif args.url:
        asin = extract_asin(args.url) or "product"
        work_dir = project_root() / "output" / "runs" / asin
    else:
        asin = "product"
        work_dir = project_root() / "output" / "runs" / asin
    return work_dir.resolve(), asin


def run_command(label: str, cmd: list[str]) -> int:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=project_root())
    if result.returncode != 0:
        print(f"Error: step failed ({label})", file=sys.stderr)
    return result.returncode


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


def print_summary(
    *,
    pipeline_path: Path,
    relevancy_path: Path,
    redundancy_path: Path,
    objectivity_path: Path,
    efficiency_path: Path,
) -> None:
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)

    if pipeline_path.exists():
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        final_alt = pipeline.get("final_alt_text") or "(none)"
        print(f"Final alt text: {final_alt}")
        print(f"Pipeline JSON:  {pipeline_path}")

    if relevancy_path.exists():
        print(f"Relevancy:      {metric_from_results(relevancy_path, 'relevancy_score')}")
    if objectivity_path.exists():
        print(f"Objectivity:    {metric_from_results(objectivity_path, 'objectivity_score')}")
    if redundancy_path.exists():
        print(f"Novelty rate:   {metric_from_results(redundancy_path, 'novelty_rate')}")
    if efficiency_path.exists():
        print(f"Efficiency:     {metric_from_results(efficiency_path, 'efficiency_per_100_words')}")


def run(args: argparse.Namespace) -> int:
    configure_stdout()

    if not args.skip_eval and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is required unless --skip-eval is set.", file=sys.stderr)
        return 1

    work_dir, asin = resolve_work_dir(args)
    work_dir.mkdir(parents=True, exist_ok=True)

    product_json = work_dir / f"{asin}.json"
    pipeline_path = project_root() / "output" / f"{asin}_pipeline.json"
    claims_dir = work_dir / "final_alt_claim_lists"
    relevancy_path = work_dir / "relevancy_evaluations.json"
    redundancy_path = work_dir / "redundancy_evaluations.json"
    objectivity_path = work_dir / "objectivity_evaluations.json"
    efficiency_path = work_dir / "efficiency_evaluations.json"

    extract_cmd = [
        sys.executable,
        "-m",
        "atsn.cli.extract_dom",
        "--output-dir",
        str(work_dir),
        "--timeout",
        str(args.extract_timeout),
    ]
    if args.url:
        extract_cmd.extend(["--url", args.url])
    else:
        extract_cmd.extend(["--html", str(resolve_path(args.html))])

    if run_command("Step 1/7 — Extract DOM + image", extract_cmd) != 0:
        return 1
    if not product_json.exists():
        print(f"Error: expected product JSON at {product_json}", file=sys.stderr)
        return 1

    pipeline_cmd = [
        sys.executable,
        "-m",
        "atsn.pipeline",
        "--product",
        str(product_json),
        "--timeout",
        str(args.timeout),
        "--delay",
        str(args.delay),
    ]
    if args.single_model:
        pipeline_cmd.extend(["--single-model", args.single_model])

    if run_command("Step 2/7 — Generate alt text (7-stage pipeline)", pipeline_cmd) != 0:
        return 1

    claims_cmd = [
        sys.executable,
        "-m",
        "atsn.evaluation.alt_to_list",
        "--source",
        "pipeline",
        "--pipeline",
        str(pipeline_path),
        "--output-dir",
        str(claims_dir),
        "--timeout",
        str(args.timeout),
        "--delay",
        str(args.delay),
    ]
    if args.single_model:
        claims_cmd.extend(["--model", args.single_model])

    if run_command("Step 3/7 — Extract claims for evaluation", claims_cmd) != 0:
        return 1

    if args.skip_eval:
        print_summary(
            pipeline_path=pipeline_path,
            relevancy_path=relevancy_path,
            redundancy_path=redundancy_path,
            objectivity_path=objectivity_path,
            efficiency_path=efficiency_path,
        )
        return 0

    eval_base = [
        "--algorithm",
        "our",
        "--products-dir",
        str(work_dir),
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
        str(work_dir),
    ]

    relevancy_cmd = [
        sys.executable,
        "-m",
        "atsn.evaluation.relevancy",
        *eval_with_image,
        "--our-claims-dir",
        str(claims_dir),
        "--output",
        str(relevancy_path),
    ]
    if run_command("Step 4/7 — Relevancy evaluation", relevancy_cmd) != 0:
        return 1

    redundancy_cmd = [
        sys.executable,
        "-m",
        "atsn.evaluation.redundancy",
        *eval_base,
        "--relevancy-input",
        str(relevancy_path),
        "--output",
        str(redundancy_path),
    ]
    if run_command("Step 5/7 — Redundancy evaluation", redundancy_cmd) != 0:
        return 1

    objectivity_cmd = [
        sys.executable,
        "-m",
        "atsn.evaluation.objectivity",
        *eval_with_image,
        "--relevancy-input",
        str(relevancy_path),
        "--output",
        str(objectivity_path),
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
        str(redundancy_path),
        "--relevancy-input",
        str(relevancy_path),
        "--output",
        str(efficiency_path),
    ]
    if run_command("Step 7/7 — Efficiency evaluation", efficiency_cmd) != 0:
        return 1

    print_summary(
        pipeline_path=pipeline_path,
        relevancy_path=relevancy_path,
        redundancy_path=redundancy_path,
        objectivity_path=objectivity_path,
        efficiency_path=efficiency_path,
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
