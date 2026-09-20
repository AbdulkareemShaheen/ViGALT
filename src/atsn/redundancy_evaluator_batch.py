#!/usr/bin/env python3
"""
Batch-run the Redundancy Evaluator on relevancy evaluation results.

Source: output/relevancy_evaluations.json (status ok)
DOM:    data/products/{product_stem}.json via build_product_dom()

Usage:
  python -m atsn.redundancy_evaluator_batch
  python -m atsn.redundancy_evaluator_batch --skip-existing
  python -m atsn.redundancy_evaluator_batch --algorithm our
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from .evaluator_batch_utils import (
    calculate_redundancy_rates,
    claims_with_relevance_from_relevancy,
    configure_stdout,
    load_existing_output,
    load_product_dom,
    load_relevancy_results,
    relevancy_job_key,
    save_output,
    summarize_results,
    utc_now_iso,
    validate_redundancy_output,
)
from .openai_backend import run_stage_openai
from .openai_config import DEFAULT_OPENAI_MODEL
from .pipeline_types import StageConfig, StageResult
from .pipeline_utils import project_root, resolve_path

PROMPTS_DIR = "prompts"
PRODUCTS_DIR = "data/products"
DEFAULT_RELEVANCY_INPUT = "output/relevancy_evaluations.json"
DEFAULT_OUTPUT_FILE = "output/redundancy_evaluations.json"
PROMPT_FILE = "evaluators/redundancy.txt"

REDUNDANCY_STAGE = StageConfig(
    name="redundancy_evaluation",
    prompt_file=PROMPT_FILE,
    model=DEFAULT_OPENAI_MODEL,
    uses_image=False,
    output_type="json",
    expected_stage=None,
)


@dataclass
class RedundancyJob:
    algorithm: str
    product_stem: str
    category: str
    input_claims: list[dict]
    product_dom: str
    metadata: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run redundancy evaluation on relevancy evaluation results."
    )
    parser.add_argument(
        "--algorithm",
        choices=["our", "asset24", "both"],
        default="both",
        help="Which algorithm results to evaluate (default: both).",
    )
    parser.add_argument(
        "--relevancy-input",
        default=DEFAULT_RELEVANCY_INPUT,
        help=f"Relevancy batch output JSON (default: {DEFAULT_RELEVANCY_INPUT}).",
    )
    parser.add_argument(
        "--products-dir",
        default=PRODUCTS_DIR,
        help=f"Product JSON folder for DOM (default: {PRODUCTS_DIR}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Combined output JSON file (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between requests (default: 1.0).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs already present in the output file with status ok.",
    )
    return parser.parse_args()


def discover_jobs(
    relevancy_input: Path,
    products_dir: Path,
    algorithm: str,
) -> list[RedundancyJob]:
    jobs: list[RedundancyJob] = []

    if algorithm in ("our", "both"):
        for result in load_relevancy_results(relevancy_input, algorithm="our"):
            jobs.append(_job_from_relevancy(result, products_dir))

    if algorithm in ("asset24", "both"):
        for result in load_relevancy_results(relevancy_input, algorithm="asset24"):
            jobs.append(_job_from_relevancy(result, products_dir))

    if not jobs:
        raise FileNotFoundError(f"No ok relevancy results found in: {relevancy_input}")
    return jobs


def _job_from_relevancy(result: dict, products_dir: Path) -> RedundancyJob:
    product_stem = str(result["product_stem"])
    return RedundancyJob(
        algorithm=str(result["algorithm"]),
        product_stem=product_stem,
        category=str(result.get("category") or ""),
        input_claims=claims_with_relevance_from_relevancy(result),
        product_dom=load_product_dom(
            product_stem,
            products_dir,
            product_file=result.get("product_file"),
        ),
        metadata={
            "relevancy_job_key": relevancy_job_key(result["algorithm"], product_stem),
            "claim_list_file": result.get("claim_list_file"),
        },
    )


def job_key(job: RedundancyJob) -> str:
    return relevancy_job_key(job.algorithm, job.product_stem)


def build_evaluation(parsed: dict, input_claims: list[dict]) -> dict:
    validate_redundancy_output(parsed, input_claims)
    claims = parsed["claims"]
    summary = calculate_redundancy_rates(claims)
    return {
        "stage": "redundancy_evaluation",
        "claims": claims,
        "summary": summary,
    }


def build_result_record(
    *,
    job: RedundancyJob,
    model: str,
    result: StageResult | None,
    processed_at: str,
    error: str | None = None,
) -> dict:
    record: dict = {
        "algorithm": job.algorithm,
        "product_stem": job.product_stem,
        "category": job.category,
        "input_claims": job.input_claims,
        "model": model,
        "prompt_file": PROMPT_FILE,
        "processed_at": processed_at,
        **job.metadata,
    }

    if result is not None:
        record["status"] = result.status
        record["duration_ms"] = result.duration_ms
        record["raw_response"] = result.raw_response
        if result.status == "ok" and isinstance(result.parsed, dict):
            try:
                record["evaluation"] = build_evaluation(result.parsed, job.input_claims)
                record["status"] = "ok"
            except ValueError as exc:
                record["status"] = "error"
                record["error"] = str(exc)
                record["evaluation"] = None
        else:
            record["evaluation"] = result.parsed
        if result.token_usage:
            record["token_usage"] = result.token_usage
        if result.error:
            record["error"] = result.error
    else:
        record["status"] = "error"
        record["error"] = error or "Unknown error"
        record["evaluation"] = None

    return record


def process_job(
    *,
    client: OpenAI,
    job: RedundancyJob,
    prompts_dir: Path,
    model: str,
    timeout_s: float,
) -> dict:
    result = run_stage_openai(
        client,
        stage=REDUNDANCY_STAGE,
        tokens={
            "PRODUCT_DOM": job.product_dom,
            "ATOMIC_CLAIMS": job.input_claims,
        },
        prompts_dir=prompts_dir,
        image_url="",
        single_model=model,
        timeout_s=timeout_s,
    )

    return build_result_record(
        job=job,
        model=model,
        result=result,
        processed_at=utc_now_iso(),
    )


def run(args: argparse.Namespace) -> int:
    configure_stdout()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.", file=sys.stderr)
        return 1

    root = project_root()
    relevancy_input = resolve_path(args.relevancy_input)
    products_dir = resolve_path(args.products_dir)
    output_path = resolve_path(args.output)
    prompts_dir = root / PROMPTS_DIR
    client = OpenAI(api_key=api_key)

    try:
        jobs = discover_jobs(relevancy_input, products_dir, args.algorithm)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    existing = load_existing_output(output_path) or {}
    existing_results = existing.get("results") if isinstance(existing.get("results"), list) else []
    existing_by_key = {
        item["job_key"]: item
        for item in existing_results
        if isinstance(item, dict) and item.get("job_key")
    }

    print("=" * 60)
    print("Redundancy Evaluator Batch")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Input:    {relevancy_input}")
    print(f"Output:   {output_path}")
    print(f"Jobs:     {len(jobs)}")
    print()

    results: list[dict] = []
    counts = {"ok": 0, "skipped": 0, "failed": 0}

    for index, job in enumerate(jobs, start=1):
        key = job_key(job)
        label = f"{job.algorithm} / {job.product_stem}"
        print(f"[{index}/{len(jobs)}] {label}")

        if args.skip_existing:
            prior = existing_by_key.get(key)
            if prior and prior.get("status") == "ok":
                print("  skipped (already ok)")
                results.append(prior)
                counts["skipped"] += 1
                continue

        try:
            record = process_job(
                client=client,
                job=job,
                prompts_dir=prompts_dir,
                model=args.model,
                timeout_s=float(args.timeout),
            )
        except Exception as exc:
            record = build_result_record(
                job=job,
                model=args.model,
                result=None,
                processed_at=utc_now_iso(),
                error=str(exc),
            )

        record["job_key"] = key
        results.append(record)

        if record.get("status") == "ok" and isinstance(record.get("evaluation"), dict):
            score = record["evaluation"].get("summary", {}).get("novelty_rate")
            print(f"  novelty_rate={score} — {record.get('duration_ms')}ms")
            counts["ok"] += 1
        else:
            print(f"  FAILED: {record.get('error')}")
            counts["failed"] += 1

        combined = {
            "stage": "redundancy_evaluation_batch",
            "processed_at": utc_now_iso(),
            "model": args.model,
            "prompt_file": PROMPT_FILE,
            "relevancy_input": str(relevancy_input),
            "total_requests": len(results),
            "summary": summarize_results(results),
            "results": results,
        }
        save_output(output_path, combined)

        if index < len(jobs):
            time.sleep(args.delay)

    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"  Succeeded: {counts['ok']}")
    print(f"  Skipped:   {counts['skipped']}")
    print(f"  Failed:    {counts['failed']}")
    print(f"  Saved to:  {output_path}")

    return 1 if counts["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
