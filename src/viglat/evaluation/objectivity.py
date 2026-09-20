#!/usr/bin/env python3
"""
Batch-run the Objectivity Evaluator on relevancy evaluation results.

Source: output/relevancy_evaluations.json (status ok)
ALT text: final_alt_text (our) or asset24_alt_text (asset24)
Image:    data/images/ with data/products/ fallback

Usage:
  python -m viglat.objectivity_evaluator_batch
  python -m viglat.objectivity_evaluator_batch --skip-existing
  python -m viglat.objectivity_evaluator_batch --algorithm asset24
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from .utils import (
    alt_text_from_relevancy,
    calculate_objectivity_score,
    configure_stdout,
    load_existing_output,
    load_relevancy_results,
    relevancy_job_key,
    resolve_image_reference,
    save_output,
    summarize_results,
    utc_now_iso,
)
from ..backend.config import DEFAULT_EVALUATION_MODEL
from ..backend.openai import run_stage_openai
from ..paths import project_root, resolve_path
from ..pipeline.types import StageConfig, StageResult

PROMPTS_DIR = "prompts"
PRODUCTS_DIR = "data/products"
DEFAULT_IMAGES_DIR = "data/images"
DEFAULT_RELEVANCY_INPUT = "output/relevancy_evaluations.json"
DEFAULT_OUTPUT_FILE = "output/objectivity_evaluations.json"
PROMPT_FILE = "evaluators/objectivity.txt"

OBJECTIVITY_STAGE = StageConfig(
    name="objectivity_evaluation",
    prompt_file=PROMPT_FILE,
    model=DEFAULT_EVALUATION_MODEL,
    uses_image=True,
    output_type="json",
    expected_stage=None,
)


@dataclass
class ObjectivityJob:
    algorithm: str
    product_stem: str
    category: str
    alt_text: str
    metadata: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run objectivity evaluation on relevancy evaluation results."
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
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help=f"Local product images folder (default: {DEFAULT_IMAGES_DIR}).",
    )
    parser.add_argument(
        "--products-dir",
        default=PRODUCTS_DIR,
        help=f"Product JSON folder for image URL fallback (default: {PRODUCTS_DIR}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Combined output JSON file (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EVALUATION_MODEL,
        help=f"OpenAI model (default: {DEFAULT_EVALUATION_MODEL}).",
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


def discover_jobs(relevancy_input: Path, algorithm: str) -> list[ObjectivityJob]:
    jobs: list[ObjectivityJob] = []

    if algorithm in ("our", "both"):
        for result in load_relevancy_results(relevancy_input, algorithm="our"):
            jobs.append(_job_from_relevancy(result))

    if algorithm in ("asset24", "both"):
        for result in load_relevancy_results(relevancy_input, algorithm="asset24"):
            jobs.append(_job_from_relevancy(result))

    if not jobs:
        raise FileNotFoundError(f"No ok relevancy results found in: {relevancy_input}")
    return jobs


def _job_from_relevancy(result: dict) -> ObjectivityJob:
    product_stem = str(result["product_stem"])
    return ObjectivityJob(
        algorithm=str(result["algorithm"]),
        product_stem=product_stem,
        category=str(result.get("category") or ""),
        alt_text=alt_text_from_relevancy(result),
        metadata={
            "relevancy_job_key": relevancy_job_key(result["algorithm"], product_stem),
            "claim_list_file": result.get("claim_list_file"),
        },
    )


def job_key(job: ObjectivityJob) -> str:
    return relevancy_job_key(job.algorithm, job.product_stem)


def build_evaluation(parsed: dict) -> dict:
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        raise ValueError("Objectivity evaluation is missing claims array.")
    summary = calculate_objectivity_score(claims)
    return {
        "stage": "objectivity_evaluation",
        "claims": claims,
        "summary": summary,
    }


def build_result_record(
    *,
    job: ObjectivityJob,
    image_reference: str,
    model: str,
    result: StageResult | None,
    processed_at: str,
    error: str | None = None,
) -> dict:
    record: dict = {
        "algorithm": job.algorithm,
        "product_stem": job.product_stem,
        "category": job.category,
        "alt_text": job.alt_text,
        "image_reference": image_reference,
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
                record["evaluation"] = build_evaluation(result.parsed)
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
    job: ObjectivityJob,
    prompts_dir: Path,
    images_dir: Path,
    products_dir: Path,
    model: str,
    timeout_s: float,
) -> dict:
    image_url, image_reference = resolve_image_reference(
        job.product_stem,
        images_dir=images_dir,
        products_dir=products_dir,
    )

    result = run_stage_openai(
        client,
        stage=OBJECTIVITY_STAGE,
        tokens={"GENERATED_ALT_TEXT": job.alt_text},
        prompts_dir=prompts_dir,
        image_url=image_url,
        single_model=model,
        timeout_s=timeout_s,
    )

    return build_result_record(
        job=job,
        image_reference=image_reference,
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
    images_dir = resolve_path(args.images_dir)
    products_dir = resolve_path(args.products_dir)
    output_path = resolve_path(args.output)
    prompts_dir = root / PROMPTS_DIR
    client = OpenAI(api_key=api_key)

    try:
        jobs = discover_jobs(relevancy_input, args.algorithm)
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
    print("Objectivity Evaluator Batch")
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
                images_dir=images_dir,
                products_dir=products_dir,
                model=args.model,
                timeout_s=float(args.timeout),
            )
        except Exception as exc:
            record = build_result_record(
                job=job,
                image_reference="",
                model=args.model,
                result=None,
                processed_at=utc_now_iso(),
                error=str(exc),
            )

        record["job_key"] = key
        results.append(record)

        if record.get("status") == "ok" and isinstance(record.get("evaluation"), dict):
            score = record["evaluation"].get("summary", {}).get("objectivity_score")
            print(f"  objectivity_score={score} — {record.get('duration_ms')}ms")
            counts["ok"] += 1
        else:
            print(f"  FAILED: {record.get('error')}")
            counts["failed"] += 1

        combined = {
            "stage": "objectivity_evaluation_batch",
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


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
