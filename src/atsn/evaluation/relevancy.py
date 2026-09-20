#!/usr/bin/env python3
"""
Batch-run the Relevancy Evaluator on claim lists from both ALT algorithms.

Sources:
  our      — output/final_alt_claim_lists/*_claims.json
  asset24  — output/ASSEST24_claim_lists/*_claims.json

Each run sends the product image, category, and atomic claims to the relevancy
prompt and writes all 60 evaluations into one combined JSON file.

Usage:
  python -m atsn.relevancy_evaluator_batch
  python -m atsn.relevancy_evaluator_batch --output output/relevancy_evaluations.json
  python -m atsn.relevancy_evaluator_batch --algorithm our
  python -m atsn.relevancy_evaluator_batch --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from .utils import (
    configure_stdout,
    load_existing_output,
    resolve_image_reference,
    save_output,
    summarize_results,
    utc_now_iso,
)
from ..backend.config import DEFAULT_OPENAI_MODEL
from ..backend.openai import run_stage_openai
from ..paths import project_root, resolve_path
from ..pipeline.types import StageConfig, StageResult

PROMPTS_DIR = "prompts"
PRODUCTS_DIR = "data/products"
DEFAULT_IMAGES_DIR = "data/images"
DEFAULT_OUR_CLAIMS_DIR = "output/final_alt_claim_lists"
DEFAULT_ASSET24_CLAIMS_DIR = "output/ASSEST24_claim_lists"
DEFAULT_OUTPUT_FILE = "output/relevancy_evaluations.json"
PROMPT_FILE = "evaluators/relevancy.txt"

RELEVANCY_STAGE = StageConfig(
    name="relevancy_evaluation",
    prompt_file=PROMPT_FILE,
    model="gemini-3.1-pro-preview",
    uses_image=True,
    output_type="json",
    expected_stage="relevancy_evaluation",
)


@dataclass
class RelevancyJob:
    algorithm: str
    claim_file: Path
    product_stem: str
    category: str
    claims: list[dict]
    metadata: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run relevancy evaluation on our and ASSEST24 claim lists."
    )
    parser.add_argument(
        "--algorithm",
        choices=["our", "asset24", "both"],
        default="both",
        help="Which claim-list source to evaluate (default: both).",
    )
    parser.add_argument(
        "--our-claims-dir",
        default=DEFAULT_OUR_CLAIMS_DIR,
        help=f"Directory with our claim lists (default: {DEFAULT_OUR_CLAIMS_DIR}).",
    )
    parser.add_argument(
        "--asset24-claims-dir",
        default=DEFAULT_ASSET24_CLAIMS_DIR,
        help=f"Directory with ASSEST24 claim lists (default: {DEFAULT_ASSET24_CLAIMS_DIR}).",
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


def category_from_stem(stem: str) -> str:
    if stem.endswith("_clothing"):
        return "CLOTHING"
    if stem.endswith("_furniture"):
        return "FURNITURE"
    raise ValueError(f"Cannot infer category from product stem: {stem!r}")


def resolve_category(product_stem: str, record: dict) -> str:
    try:
        return category_from_stem(product_stem)
    except ValueError:
        pass

    category = record.get("classification_category")
    if category in ("CLOTHING", "FURNITURE"):
        return str(category)

    source = record.get("source_pipeline") or record.get("source")
    if source:
        pipeline_path = Path(str(source))
        if pipeline_path.exists():
            pipeline_record = json.loads(pipeline_path.read_text(encoding="utf-8"))
            category = pipeline_record.get("classification_category")
            if category in ("CLOTHING", "FURNITURE"):
                return str(category)

    raise ValueError(
        f"Cannot infer category for product stem {product_stem!r}. "
        "Expected *_clothing/*_furniture stem or classification_category in claim/pipeline JSON."
    )


def category_from_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "clothing":
        return "CLOTHING"
    if normalized == "furniture":
        return "FURNITURE"
    raise ValueError(f"Unknown product type: {value!r}")


def product_stem_for_index(index: int, products_dir: Path) -> str:
    for suffix in ("clothing", "furniture"):
        path = products_dir / f"{index}_{suffix}.json"
        if path.exists():
            return path.stem
    raise FileNotFoundError(f"No product JSON found for index {index} in {products_dir}")


def load_claims_from_file(claim_file: Path) -> list[dict]:
    record = json.loads(claim_file.read_text(encoding="utf-8"))
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        raise ValueError(f"{claim_file} is missing parsed claim extraction output.")

    claims = parsed.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError(f"{claim_file} has no claims in parsed output.")

    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError(f"{claim_file} contains a non-object claim entry.")
        if not claim.get("claim_id") or not claim.get("claim"):
            raise ValueError(f"{claim_file} contains a claim missing claim_id or claim text.")

    return claims


def discover_our_jobs(claims_dir: Path, products_dir: Path) -> list[RelevancyJob]:
    if not claims_dir.exists():
        raise FileNotFoundError(f"Our claim-list directory not found: {claims_dir}")

    jobs: list[RelevancyJob] = []
    for claim_file in sorted(claims_dir.glob("*_claims.json")):
        record = json.loads(claim_file.read_text(encoding="utf-8"))
        product_stem = str(record.get("product_stem") or claim_file.stem.removesuffix("_claims"))
        category = resolve_category(product_stem, record)
        claims = load_claims_from_file(claim_file)
        metadata = {
            "source_pipeline": record.get("source_pipeline"),
            "final_alt_text": record.get("final_alt_text"),
        }
        if record.get("product_file"):
            metadata["product_file"] = record.get("product_file")
        if record.get("classification_category"):
            metadata["classification_category"] = record.get("classification_category")
        jobs.append(
            RelevancyJob(
                algorithm="our",
                claim_file=claim_file.resolve(),
                product_stem=product_stem,
                category=category,
                claims=claims,
                metadata=metadata,
            )
        )

    if not jobs:
        raise FileNotFoundError(f"No claim files found in: {claims_dir}")
    return jobs


def _asset24_sort_key(path: Path) -> int:
    return int(path.stem.split("_", 1)[0])


def discover_asset24_jobs(claims_dir: Path, products_dir: Path) -> list[RelevancyJob]:
    if not claims_dir.exists():
        raise FileNotFoundError(f"ASSEST24 claim-list directory not found: {claims_dir}")

    jobs: list[RelevancyJob] = []
    for claim_file in sorted(claims_dir.glob("*_claims.json"), key=_asset24_sort_key):
        record = json.loads(claim_file.read_text(encoding="utf-8"))
        index = record.get("index")
        if index is None:
            index = int(claim_file.stem.split("_", 1)[0])

        product_stem = product_stem_for_index(int(index), products_dir)
        type_value = record.get("type")
        category = category_from_type(type_value) if type_value else category_from_stem(product_stem)
        claims = load_claims_from_file(claim_file)
        jobs.append(
            RelevancyJob(
                algorithm="asset24",
                claim_file=claim_file.resolve(),
                product_stem=product_stem,
                category=category,
                claims=claims,
                metadata={
                    "index": int(index),
                    "excel_row": record.get("excel_row"),
                    "asset24_alt_text": record.get("asset24_alt_text") or record.get("alt_text"),
                },
            )
        )

    if not jobs:
        raise FileNotFoundError(f"No claim files found in: {claims_dir}")
    return jobs


def job_key(job: RelevancyJob) -> str:
    return f"{job.algorithm}:{job.claim_file.name}"


def build_result_record(
    *,
    job: RelevancyJob,
    image_reference: str,
    model: str,
    result: StageResult | None,
    processed_at: str,
    error: str | None = None,
) -> dict:
    record: dict = {
        "algorithm": job.algorithm,
        "claim_list_file": str(job.claim_file),
        "product_stem": job.product_stem,
        "category": job.category,
        "image_reference": image_reference,
        "input_claims": job.claims,
        "model": model,
        "prompt_file": PROMPT_FILE,
        "processed_at": processed_at,
        **job.metadata,
    }

    if result is not None:
        record["status"] = result.status
        record["duration_ms"] = result.duration_ms
        record["raw_response"] = result.raw_response
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
    job: RelevancyJob,
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
        stage=RELEVANCY_STAGE,
        tokens={
            "CATEGORY": job.category,
            "ATOMIC_CLAIMS": job.claims,
        },
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
    our_claims_dir = resolve_path(args.our_claims_dir)
    asset24_claims_dir = resolve_path(args.asset24_claims_dir)
    images_dir = resolve_path(args.images_dir)
    products_dir = resolve_path(args.products_dir)
    output_path = resolve_path(args.output)
    prompts_dir = root / PROMPTS_DIR
    client = OpenAI(api_key=api_key)

    jobs: list[RelevancyJob] = []
    try:
        if args.algorithm in ("our", "both"):
            jobs.extend(discover_our_jobs(our_claims_dir, products_dir))
        if args.algorithm in ("asset24", "both"):
            jobs.extend(discover_asset24_jobs(asset24_claims_dir, products_dir))
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
    print("Relevancy Evaluator Batch")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Output:   {output_path}")
    print(f"Jobs:     {len(jobs)}")
    print()

    results: list[dict] = []
    counts = {"ok": 0, "skipped": 0, "failed": 0}

    for index, job in enumerate(jobs, start=1):
        key = job_key(job)
        label = f"{job.algorithm} / {job.claim_file.name} / {job.product_stem}"
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
            score = record["evaluation"].get("summary", {}).get("relevancy_score")
            print(f"  score={score} — {record.get('duration_ms')}ms")
            counts["ok"] += 1
        else:
            print(f"  FAILED: {record.get('error')}")
            counts["failed"] += 1

        combined = {
            "stage": "relevancy_evaluation_batch",
            "processed_at": utc_now_iso(),
            "model": args.model,
            "prompt_file": PROMPT_FILE,
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
