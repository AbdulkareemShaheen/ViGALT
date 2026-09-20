#!/usr/bin/env python3
"""
Multi-stage ALT-text pipeline.

Runs 7 stages (classifier → generator → claim extraction → 3 validators → fuser)
via the OpenAI Chat Completions API.

Usage:
  python -m atsn.pipeline                          # first product only (validation)
  python -m atsn.pipeline --product data/products/1_clothing.json
  python -m atsn.pipeline --all
  python -m atsn.pipeline --single-model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .openai_backend import run_stage_openai
from .openai_config import resolve_openai_model
from .pipeline_types import StageConfig, StageResult
from .pipeline_utils import (
    ProductData,
    list_product_files,
    load_product,
    project_root,
    resolve_path,
)

OUTPUT_DIR = "output"
PROMPTS_DIR = "prompts"
PRODUCTS_DIR = "data/products"
DOWNLOADS_DIR = "downloads"

GENERATOR_PROMPTS = {
    "CLOTHING": "clothing_generator.txt",
    "FURNITURE": "furniture_generator.txt",
}

VALID_CATEGORIES = frozenset(GENERATOR_PROMPTS)

STAGE_ORDER: list[str] = [
    "classification",
    "generation",
    "claim_extraction",
    "accuracy_validation",
    "completeness_validation",
    "redundancy_validation",
    "fusion",
]

STAGES: list[StageConfig] = [
    StageConfig(
        name="classification",
        prompt_file="classifier.txt",
        model="gemini-3.5-flash-lite",
        uses_image=True,
        output_type="json",
        expected_stage="classification",
    ),
    StageConfig(
        name="claim_extraction",
        prompt_file="alt_to_list.txt",
        model="gemini-3.5-flash-lite",
        uses_image=False,
        output_type="json",
        expected_stage="claim_extraction",
    ),
    StageConfig(
        name="accuracy_validation",
        prompt_file="accuracy_validator.txt",
        model="gemini-3.1-pro-preview",
        uses_image=True,
        output_type="json",
        expected_stage="accuracy_validation",
    ),
    StageConfig(
        name="completeness_validation",
        prompt_file="completeness_validator.txt",
        model="gemini-3.1-pro-preview",
        uses_image=True,
        output_type="json",
        expected_stage="completeness_validation",
    ),
    StageConfig(
        name="redundancy_validation",
        prompt_file="redundancy_validator.txt",
        model="gemini-3.7-flash",
        uses_image=True,
        output_type="json",
        expected_stage="redundancy_validation",
    ),
    StageConfig(
        name="fusion",
        prompt_file="fuser.txt",
        model="gemini-3.7-flash",
        uses_image=False,
        output_type="json",
        expected_stage="fusion",
    ),
]

GENERATION_STAGE = StageConfig(
    name="generation",
    prompt_file="",
    model="gemini-3.1-pro-preview",
    uses_image=True,
    output_type="text",
)

CONTINUABLE_STAGES = frozenset({"accuracy_validation", "completeness_validation", "redundancy_validation"})


class PipelineState:
    def __init__(self, product: ProductData) -> None:
        self.product = product
        self.classification: dict | None = None
        self.alt_text: str | None = None
        self.atomic_claims: list | None = None
        self.accuracy_output: dict | None = None
        self.completeness_output: dict | None = None
        self.redundancy_output: dict | None = None
        self.final_alt_text: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 7-stage ALT-text pipeline via OpenAI API."
    )
    parser.add_argument(
        "--product",
        default=None,
        help="Single product JSON file (default: first product in data/products/).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all product JSON files in data/products/.",
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
        help="Per-stage response timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between stages (default: 3.0).",
    )
    parser.add_argument(
        "--from-stage",
        choices=STAGE_ORDER,
        default=None,
        help="Resume from this stage, reusing prior stage outputs from the saved pipeline JSON.",
    )
    return parser.parse_args()


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_products(args: argparse.Namespace) -> list[Path]:
    products_dir = project_root() / PRODUCTS_DIR
    if args.all:
        return list_product_files(products_dir)
    if args.product:
        return [resolve_path(args.product)]
    return [list_product_files(products_dir)[0]]


def output_path_for(product_file: Path) -> Path:
    return project_root() / OUTPUT_DIR / f"{product_file.stem}_pipeline.json"


def build_pipeline_record(
    product: ProductData,
    *,
    started_at: str,
    completed_at: str | None = None,
    classification_category: str | None = None,
    final_alt_text: str | None = None,
    stages: dict[str, dict] | None = None,
) -> dict:
    return {
        "product_file": str(product.product_file),
        "image_url": product.image_url,
        "started_at": started_at,
        "completed_at": completed_at,
        "classification_category": classification_category,
        "final_alt_text": final_alt_text,
        "stages": stages or {},
    }


def save_pipeline_output(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Pipeline output saved: {path}")


def stage_result_to_dict(result: StageResult) -> dict:
    payload = {
        "model": result.model,
        "prompt_file": result.prompt_file,
        "raw_response": result.raw_response,
        "parsed": result.parsed,
        "duration_ms": result.duration_ms,
        "status": result.status,
        "backend": result.backend,
    }
    if result.error:
        payload["error"] = result.error
    if result.token_usage:
        payload["token_usage"] = result.token_usage
    return payload


def build_tokens(state: PipelineState, stage_name: str) -> dict[str, str | dict | list]:
    product = state.product
    if stage_name in ("classification", "generation"):
        return {"SURROUNDING_TEXT": product.surrounding_text}
    if stage_name == "claim_extraction":
        if not state.alt_text:
            raise RuntimeError("ALT text missing for claim extraction stage.")
        return {"ALT_TEXT": state.alt_text}
    if stage_name == "accuracy_validation":
        if not state.atomic_claims:
            raise RuntimeError("Atomic claims missing for accuracy validation.")
        return {"ATOMIC_CLAIMS": state.atomic_claims}
    if stage_name == "completeness_validation":
        if not state.atomic_claims:
            raise RuntimeError("Atomic claims missing for completeness validation.")
        return {"ATOMIC_CLAIMS": state.atomic_claims}
    if stage_name == "redundancy_validation":
        if not state.atomic_claims:
            raise RuntimeError("Atomic claims missing for redundancy validation.")
        return {
            "PRODUCT_DOM": product.product_dom,
            "ATOMIC_CLAIMS": state.atomic_claims,
        }
    if stage_name == "fusion":
        missing = [
            name
            for name, value in [
                ("ALT_TEXT", state.alt_text),
                ("ATOMIC_CLAIMS", state.atomic_claims),
                ("ACCURACY_OUTPUT", state.accuracy_output),
                ("COMPLETENESS_OUTPUT", state.completeness_output),
                ("REDUNDANCY_OUTPUT", state.redundancy_output),
            ]
            if value is None
        ]
        if missing:
            raise RuntimeError(f"Missing inputs for fuser: {', '.join(missing)}")
        return {
            "ALT_TEXT": state.alt_text,
            "ATOMIC_CLAIMS": state.atomic_claims,
            "ACCURACY_OUTPUT": state.accuracy_output,
            "COMPLETENESS_OUTPUT": state.completeness_output,
            "REDUNDANCY_OUTPUT": state.redundancy_output,
        }
    raise ValueError(f"Unknown stage: {stage_name}")


def load_pipeline_record(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline output not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def hydrate_state_from_record(state: PipelineState, record: dict) -> None:
    stages = record.get("stages") or {}

    classification = stages.get("classification", {}).get("parsed")
    if isinstance(classification, dict):
        state.classification = classification

    generation = stages.get("generation", {}).get("parsed")
    if isinstance(generation, str) and generation.strip():
        state.alt_text = generation

    claim_extraction = stages.get("claim_extraction", {}).get("parsed")
    if isinstance(claim_extraction, dict):
        claims = claim_extraction.get("claims")
        if isinstance(claims, list) and claims:
            state.atomic_claims = claims

    accuracy = stages.get("accuracy_validation", {}).get("parsed")
    if isinstance(accuracy, dict):
        state.accuracy_output = accuracy

    completeness = stages.get("completeness_validation", {}).get("parsed")
    if isinstance(completeness, dict):
        state.completeness_output = completeness

    redundancy = stages.get("redundancy_validation", {}).get("parsed")
    if isinstance(redundancy, dict):
        state.redundancy_output = redundancy

    fusion = stages.get("fusion", {}).get("parsed")
    if isinstance(fusion, dict):
        final = fusion.get("final_alt_text")
        if isinstance(final, str) and final.strip():
            state.final_alt_text = final.strip()


def stage_config_for(name: str) -> StageConfig:
    if name == "generation":
        raise ValueError("Use generation_stage_for() for the generation stage.")
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise ValueError(f"Unknown stage: {name}")


def stages_from(from_stage: str) -> list[StageConfig]:
    if from_stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage: {from_stage}")
    start = STAGE_ORDER.index(from_stage)
    ordered: list[StageConfig] = []
    for name in STAGE_ORDER[start:]:
        if name == "generation":
            raise RuntimeError("Cannot resume from generation without classification state.")
        ordered.append(stage_config_for(name))
    return ordered


def apply_stage_output(state: PipelineState, stage: StageConfig, parsed: dict | str | None) -> None:
    if stage.name == "classification":
        if not isinstance(parsed, dict):
            raise ValueError("Classification stage must return JSON.")
        category = parsed.get("category")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Unexpected classification category: {category!r}. "
                f"Expected one of: {sorted(VALID_CATEGORIES)}"
            )
        state.classification = parsed
    elif stage.name == "generation":
        state.alt_text = parsed if isinstance(parsed, str) else str(parsed)
    elif stage.name == "claim_extraction":
        if not isinstance(parsed, dict):
            raise ValueError("Claim extraction stage must return JSON.")
        claims = parsed.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError("Claim extraction returned no claims.")
        state.atomic_claims = claims
    elif stage.name == "accuracy_validation":
        state.accuracy_output = parsed if isinstance(parsed, dict) else None
    elif stage.name == "completeness_validation":
        state.completeness_output = parsed if isinstance(parsed, dict) else None
    elif stage.name == "redundancy_validation":
        state.redundancy_output = parsed if isinstance(parsed, dict) else None
    elif stage.name == "fusion":
        if not isinstance(parsed, dict):
            raise ValueError("Fuser stage must return JSON.")
        final = parsed.get("final_alt_text")
        if not final or not isinstance(final, str):
            raise ValueError("Fuser output missing final_alt_text.")
        state.final_alt_text = final.strip()


def display_model(stage: StageConfig, single_model: str | None) -> str:
    return resolve_openai_model(stage.name, single_model)


def execute_stage(
    client: OpenAI,
    *,
    stage: StageConfig,
    tokens: dict,
    prompts_dir: Path,
    product: ProductData,
    timeout_s: float,
    single_model: str | None = None,
) -> StageResult:
    return run_stage_openai(
        client,
        stage=stage,
        tokens=tokens,
        prompts_dir=prompts_dir,
        image_url=product.image_url,
        single_model=single_model,
        timeout_s=timeout_s,
    )


def should_stop_pipeline(stage_name: str, result: StageResult) -> bool:
    if result.status == "ok":
        return False
    if result.status == "parse_error" and stage_name in CONTINUABLE_STAGES:
        return False
    return True


def stage_delay(delay_s: float) -> None:
    time.sleep(delay_s)


def generation_stage_for(category: str) -> StageConfig:
    prompt_file = GENERATOR_PROMPTS[category]
    return StageConfig(
        name=GENERATION_STAGE.name,
        prompt_file=prompt_file,
        model=GENERATION_STAGE.model,
        uses_image=GENERATION_STAGE.uses_image,
        output_type=GENERATION_STAGE.output_type,
    )


def run_pipeline_for_product(
    *,
    client: OpenAI,
    product_file: Path,
    prompts_dir: Path,
    timeout_s: float,
    delay_s: float,
    from_stage: str | None = None,
    single_model: str | None = None,
) -> dict:
    product = load_product(product_file, project_root() / DOWNLOADS_DIR)
    state = PipelineState(product)
    out_path = output_path_for(product_file)

    if from_stage:
        prior = load_pipeline_record(out_path)
        hydrate_state_from_record(state, prior)
        resume_idx = STAGE_ORDER.index(from_stage)
        for name in STAGE_ORDER[resume_idx:]:
            if name == "completeness_validation":
                state.completeness_output = None
            elif name == "redundancy_validation":
                state.redundancy_output = None
            elif name == "fusion":
                state.final_alt_text = None
        started_at = prior.get("started_at") or utc_now_iso()
        stages_out = {
            name: data
            for name, data in (prior.get("stages") or {}).items()
            if STAGE_ORDER.index(name) < resume_idx
        }
        record = build_pipeline_record(
            product,
            started_at=started_at,
            classification_category=prior.get("classification_category"),
            stages=stages_out,
        )
        record["completed_at"] = None
        record["final_alt_text"] = None
    else:
        started_at = utc_now_iso()
        stages_out = {}
        record = build_pipeline_record(product, started_at=started_at, stages=stages_out)

    save_pipeline_output(out_path, record)

    print("\n" + "=" * 60)
    print(f"Pipeline: {product_file.name}")
    print(f"Image:    {product.image_path.name}")
    if from_stage:
        print(f"Resume:   from {from_stage}")
    print("=" * 60)

    pipeline_stopped = False

    def run_one_stage(stage: StageConfig) -> None:
        nonlocal pipeline_stopped
        if pipeline_stopped:
            return

        model_label = display_model(stage, single_model)
        print(f"\n--- Stage: {stage.name} ({model_label}) ---")
        tokens = build_tokens(state, stage.name)
        result = execute_stage(
            client,
            stage=stage,
            tokens=tokens,
            prompts_dir=prompts_dir,
            product=product,
            timeout_s=timeout_s,
            single_model=single_model,
        )
        stages_out[stage.name] = stage_result_to_dict(result)
        record["stages"] = stages_out
        save_pipeline_output(out_path, record)

        if should_stop_pipeline(stage.name, result):
            pipeline_stopped = True
            raise RuntimeError(
                f"Stage {stage.name} failed ({result.status}): {result.error}"
            )

        if result.status == "ok":
            apply_stage_output(state, stage, result.parsed)

        if stage.name == "classification" and state.classification:
            record["classification_category"] = state.classification["category"]
        if stage.name == "fusion" and state.final_alt_text:
            record["final_alt_text"] = state.final_alt_text

        save_pipeline_output(out_path, record)
        stage_delay(delay_s)

    if from_stage:
        for stage in stages_from(from_stage):
            run_one_stage(stage)
    else:
        run_one_stage(STAGES[0])

        category = state.classification["category"]  # type: ignore[index]
        run_one_stage(generation_stage_for(category))

        for stage in STAGES[1:]:
            run_one_stage(stage)

    record["completed_at"] = utc_now_iso()
    record["final_alt_text"] = state.final_alt_text
    save_pipeline_output(out_path, record)
    return record


def run_openai_pipeline(args: argparse.Namespace, product_files: list[Path]) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.", file=sys.stderr)
        return 1

    root = project_root()
    prompts_dir = root / PROMPTS_DIR
    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("Multi-Stage OpenAI Pipeline")
    print("=" * 60)
    print(f"Products: {len(product_files)}")
    if args.single_model:
        print(f"Model:    {args.single_model} (all stages)")
    for path in product_files:
        print(f"  - {path.name}")
    print()

    results: list[dict] = []
    try:
        for product_file in product_files:
            record = run_pipeline_for_product(
                client=client,
                product_file=product_file,
                prompts_dir=prompts_dir,
                timeout_s=float(args.timeout),
                delay_s=args.delay,
                from_stage=args.from_stage,
                single_model=args.single_model,
            )
            results.append(record)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        for record in results:
            out = output_path_for(Path(record["product_file"]))
            final = record.get("final_alt_text") or "(none)"
            print(f"  {out}")
            print(f"    final_alt_text: {final[:80]}...")
        return 0
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    configure_stdout()
    product_files = resolve_products(args)
    return run_openai_pipeline(args, product_files)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
