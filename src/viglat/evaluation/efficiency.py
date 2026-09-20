#!/usr/bin/env python3
"""
Batch-run the Efficiency Evaluator (deterministic, no LLM).

Formula: (relevant_novel_claims / ALT word count) * 100

Sources:
  redundancy — output/redundancy_evaluations.json
  alt text   — embedded in redundancy records or relevancy input

Usage:
  python -m viglat.efficiency_evaluator_batch
  python -m viglat.efficiency_evaluator_batch --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .utils import (
    alt_text_from_relevancy,
    calculate_efficiency,
    configure_stdout,
    count_alt_words,
    load_existing_output,
    load_relevancy_results,
    relevancy_job_key,
    save_output,
    summarize_results,
    utc_now_iso,
)
from ..paths import resolve_path

DEFAULT_REDUNDANCY_INPUT = "output/redundancy_evaluations.json"
DEFAULT_RELEVANCY_INPUT = "output/relevancy_evaluations.json"
DEFAULT_OUTPUT_FILE = "output/efficiency_evaluations.json"


@dataclass
class EfficiencyJob:
    algorithm: str
    product_stem: str
    category: str
    alt_text: str
    relevant_novel_claims: int
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run efficiency evaluation from redundancy batch results."
    )
    parser.add_argument(
        "--algorithm",
        choices=["our", "asset24", "both"],
        default="both",
        help="Which algorithm results to evaluate (default: both).",
    )
    parser.add_argument(
        "--redundancy-input",
        default=DEFAULT_REDUNDANCY_INPUT,
        help=f"Redundancy batch output JSON (default: {DEFAULT_REDUNDANCY_INPUT}).",
    )
    parser.add_argument(
        "--relevancy-input",
        default=DEFAULT_RELEVANCY_INPUT,
        help=f"Relevancy batch output for ALT text fallback (default: {DEFAULT_RELEVANCY_INPUT}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Combined output JSON file (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs already present in the output file with status ok.",
    )
    return parser.parse_args()


def load_redundancy_results(path: Path, algorithm: str | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Redundancy evaluations not found: {path}")

    record = json.loads(path.read_text(encoding="utf-8"))
    results = record.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} is missing a results array.")

    filtered: list[dict] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            continue
        if not isinstance(item.get("evaluation"), dict):
            continue
        if algorithm and item.get("algorithm") != algorithm:
            continue
        filtered.append(item)
    return filtered


def build_alt_text_lookup(relevancy_input: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for result in load_relevancy_results(relevancy_input):
        key = relevancy_job_key(result["algorithm"], result["product_stem"])
        lookup[key] = alt_text_from_relevancy(result)
    return lookup


def discover_jobs(
    redundancy_input: Path,
    relevancy_input: Path,
    algorithm: str,
) -> list[EfficiencyJob]:
    alt_lookup = build_alt_text_lookup(relevancy_input)
    jobs: list[EfficiencyJob] = []

    algorithms = ["our", "asset24"] if algorithm == "both" else [algorithm]
    for algo in algorithms:
        for result in load_redundancy_results(redundancy_input, algorithm=algo):
            product_stem = str(result["product_stem"])
            key = relevancy_job_key(algo, product_stem)

            evaluation = result["evaluation"]
            summary = evaluation.get("summary")
            if not isinstance(summary, dict):
                raise ValueError(f"Redundancy result for {key} is missing summary.")

            relevant_novel = summary.get("relevant_novel_claims")
            if relevant_novel is None:
                raise ValueError(f"Redundancy result for {key} is missing relevant_novel_claims.")

            alt_text = result.get("alt_text") or alt_lookup.get(key)
            if not alt_text:
                raise ValueError(f"No ALT text found for {key}.")

            jobs.append(
                EfficiencyJob(
                    algorithm=algo,
                    product_stem=product_stem,
                    category=str(result.get("category") or ""),
                    alt_text=str(alt_text),
                    relevant_novel_claims=int(relevant_novel),
                    metadata={
                        "redundancy_job_key": key,
                        "claim_list_file": result.get("claim_list_file"),
                    },
                )
            )

    if not jobs:
        raise FileNotFoundError(f"No ok redundancy results found in: {redundancy_input}")
    return jobs


def job_key(job: EfficiencyJob) -> str:
    return relevancy_job_key(job.algorithm, job.product_stem)


def process_job(job: EfficiencyJob) -> dict:
    word_count = count_alt_words(job.alt_text)
    efficiency = calculate_efficiency(job.relevant_novel_claims, word_count)

    evaluation = {
        "stage": "efficiency_evaluation",
        "metric": "Relevant Novel Claim Efficiency",
        "formula": "(Relevant AND Novel claims / ALT word count) * 100",
        "unit": "relevant novel claims per 100 words",
        "summary": {
            "relevant_novel_claims": job.relevant_novel_claims,
            "word_count": word_count,
            "efficiency_per_100_words": efficiency,
        },
    }

    return {
        "algorithm": job.algorithm,
        "product_stem": job.product_stem,
        "category": job.category,
        "alt_text": job.alt_text,
        "processed_at": utc_now_iso(),
        "status": "ok",
        "evaluation": evaluation,
        **job.metadata,
    }


def run(args: argparse.Namespace) -> int:
    configure_stdout()

    redundancy_input = resolve_path(args.redundancy_input)
    relevancy_input = resolve_path(args.relevancy_input)
    output_path = resolve_path(args.output)

    try:
        jobs = discover_jobs(redundancy_input, relevancy_input, args.algorithm)
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
    print("Efficiency Evaluator Batch")
    print("=" * 60)
    print(f"Input:    {redundancy_input}")
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
            record = process_job(job)
        except Exception as exc:
            record = {
                "algorithm": job.algorithm,
                "product_stem": job.product_stem,
                "category": job.category,
                "processed_at": utc_now_iso(),
                "status": "error",
                "error": str(exc),
                "evaluation": None,
                **job.metadata,
            }
            counts["failed"] += 1
            print(f"  FAILED: {exc}")
        else:
            counts["ok"] += 1
            score = record["evaluation"]["summary"]["efficiency_per_100_words"]
            print(f"  efficiency={score:.2f} per 100 words")

        record["job_key"] = key
        results.append(record)

        combined = {
            "stage": "efficiency_evaluation_batch",
            "processed_at": utc_now_iso(),
            "redundancy_input": str(redundancy_input),
            "relevancy_input": str(relevancy_input),
            "total_requests": len(results),
            "summary": summarize_results(results),
            "results": results,
        }
        save_output(output_path, combined)

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
