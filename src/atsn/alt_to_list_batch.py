#!/usr/bin/env python3
"""
Batch-run claim extraction on ALT texts via GPT 5.6 and alt_to_list.txt.

Sources:
  pipeline — final_alt_text from output/*_pipeline.json
  asset24  — ASSEST24 column from data/ATSN_Dataset.xlsx

Usage:
  python -m atsn.alt_to_list_batch --source pipeline --all
  python -m atsn.alt_to_list_batch --source asset24 --all
  python -m atsn.alt_to_list_batch --source pipeline --pipeline output/1_clothing_pipeline.json
  python -m atsn.alt_to_list_batch --source asset24 --index 5
  python -m atsn.alt_to_list_batch --source asset24 --all --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
from openai import OpenAI

from .openai_backend import run_stage_openai
from .openai_config import DEFAULT_OPENAI_MODEL
from .pipeline_types import StageConfig, StageResult
from .pipeline_utils import project_root, resolve_path

PROMPTS_DIR = "prompts"
DEFAULT_PIPELINE_DIR = "output"
DEFAULT_PIPELINE_OUTPUT_DIR = "output/final_alt_claim_lists"
DEFAULT_ASSET24_EXCEL = "data/ATSN_Dataset.xlsx"
DEFAULT_ASSET24_OUTPUT_DIR = "output/ASSEST24_claim_lists"
ASSET24_COLUMN = "ASSEST24 GENERATED SHORT CONTEXT-Aware ALT TEXT"
PROMPT_FILE = "alt_to_list.txt"

CLAIM_EXTRACTION_STAGE = StageConfig(
    name="claim_extraction",
    prompt_file=PROMPT_FILE,
    model="gemini-3.5-flash-lite",
    uses_image=False,
    output_type="json",
    expected_stage="claim_extraction",
)


@dataclass
class AltJob:
    stem: str
    alt_text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ALT-to-List claim extraction on ALT texts via OpenAI."
    )
    parser.add_argument(
        "--source",
        choices=["pipeline", "asset24"],
        default="pipeline",
        help="ALT text source: pipeline JSON outputs or ATSN Dataset.xlsx (default: pipeline).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all items from the selected source.",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Single pipeline JSON file (pipeline source only).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help="Process one item by 1-based index (asset24 source only).",
    )
    parser.add_argument(
        "--pipeline-dir",
        default=DEFAULT_PIPELINE_DIR,
        help=f"Directory containing *_pipeline.json files (default: {DEFAULT_PIPELINE_DIR}).",
    )
    parser.add_argument(
        "--excel-file",
        default=DEFAULT_ASSET24_EXCEL,
        help=f"ATSN Dataset workbook (default: {DEFAULT_ASSET24_EXCEL}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults by source: final_alt_claim_lists or ASSEST24_claim_lists).",
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
        help="Skip items whose output file already exists with status ok.",
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


def default_output_dir(source: str) -> str:
    if source == "asset24":
        return DEFAULT_ASSET24_OUTPUT_DIR
    return DEFAULT_PIPELINE_OUTPUT_DIR


def discover_pipeline_files(pipeline_dir: Path) -> list[Path]:
    if not pipeline_dir.exists():
        raise FileNotFoundError(f"Pipeline directory not found: {pipeline_dir}")
    files = sorted(
        path.resolve()
        for path in pipeline_dir.glob("*_pipeline.json")
        if path.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No *_pipeline.json files found in: {pipeline_dir}")
    return files


def stem_from_pipeline(path: Path) -> str:
    name = path.name
    if name.endswith("_pipeline.json"):
        return name[: -len("_pipeline.json")]
    return path.stem


def load_final_alt_text(pipeline_file: Path) -> str:
    record = json.loads(pipeline_file.read_text(encoding="utf-8"))
    final_alt = record.get("final_alt_text")
    if not isinstance(final_alt, str) or not final_alt.strip():
        raise ValueError(f"Missing or empty final_alt_text in {pipeline_file}")
    return final_alt.strip()


def _cell_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_asset24_jobs(excel_file: Path) -> list[AltJob]:
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")

    wb = openpyxl.load_workbook(excel_file, read_only=True, data_only=True)
    ws = wb.active
    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
    if not header_row:
        raise ValueError(f"No header row in {excel_file}")

    headers = [_cell_str(cell) for cell in header_row]
    try:
        alt_col = headers.index(ASSET24_COLUMN)
    except ValueError as exc:
        raise ValueError(
            f"Column {ASSET24_COLUMN!r} not found in {excel_file}. "
            f"Available: {headers[:8]}"
        ) from exc

    type_col = headers.index("Type") if "Type" in headers else None
    url_col = headers.index("URL") if "URL" in headers else None
    main_alt_col = headers.index("Main Image Alt") if "Main Image Alt" in headers else None

    jobs: list[AltJob] = []
    for excel_row, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if row is None:
            continue
        alt_text = _cell_str(row[alt_col]) if len(row) > alt_col else ""
        if not alt_text:
            continue

        index = len(jobs) + 1
        stem = f"{index:02d}"
        metadata = {"excel_row": excel_row, "index": index}
        if type_col is not None and len(row) > type_col:
            metadata["type"] = _cell_str(row[type_col])
        if url_col is not None and len(row) > url_col:
            metadata["url"] = _cell_str(row[url_col])
        if main_alt_col is not None and len(row) > main_alt_col:
            metadata["main_image_alt"] = _cell_str(row[main_alt_col])

        jobs.append(
            AltJob(
                stem=stem,
                alt_text=alt_text,
                source=str(excel_file),
                metadata=metadata,
            )
        )

    wb.close()
    if not jobs:
        raise ValueError(f"No ASSEST24 ALT texts found in column {ASSET24_COLUMN!r}")
    return jobs


def resolve_pipeline_jobs(args: argparse.Namespace) -> list[AltJob]:
    if args.pipeline:
        pipeline_files = [resolve_path(args.pipeline)]
    else:
        pipeline_files = discover_pipeline_files(resolve_path(args.pipeline_dir))

    return [
        AltJob(
            stem=stem_from_pipeline(pipeline_file),
            alt_text="",
            source=str(pipeline_file),
            metadata={"product_stem": stem_from_pipeline(pipeline_file)},
        )
        for pipeline_file in pipeline_files
    ]


def resolve_jobs(args: argparse.Namespace) -> list[AltJob]:
    if args.source == "asset24":
        jobs = load_asset24_jobs(resolve_path(args.excel_file))
        if args.index is not None:
            if args.index < 1 or args.index > len(jobs):
                raise ValueError(f"--index must be between 1 and {len(jobs)}")
            return [jobs[args.index - 1]]
        if not args.all:
            raise ValueError("asset24 source requires --all or --index N")
        return jobs

    if args.index is not None:
        raise ValueError("--index is only supported with --source asset24")

    if args.pipeline or args.all:
        return resolve_pipeline_jobs(args)
    raise ValueError("pipeline source requires --all or --pipeline")


def output_path_for(stem: str, output_dir: Path) -> Path:
    return output_dir / f"{stem}_claims.json"


def load_existing_output(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_output(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_record(
    *,
    job: AltJob,
    source_kind: str,
    model: str,
    result: StageResult | None,
    processed_at: str,
    error: str | None = None,
) -> dict:
    record: dict = {
        "source_kind": source_kind,
        "source": job.source,
        "stem": job.stem,
        "alt_text": job.alt_text,
        "model": model,
        "prompt_file": PROMPT_FILE,
        "processed_at": processed_at,
        **job.metadata,
    }
    if source_kind == "pipeline":
        record["product_stem"] = job.metadata.get("product_stem", job.stem)
        record["final_alt_text"] = job.alt_text
    if source_kind == "asset24":
        record["asset24_alt_text"] = job.alt_text

    if result is not None:
        record["status"] = result.status
        record["duration_ms"] = result.duration_ms
        record["raw_response"] = result.raw_response
        record["parsed"] = result.parsed
        if result.token_usage:
            record["token_usage"] = result.token_usage
        if result.error:
            record["error"] = result.error
    else:
        record["status"] = "error"
        record["error"] = error or "Unknown error"
    return record


def process_one(
    *,
    client: OpenAI,
    job: AltJob,
    source_kind: str,
    output_dir: Path,
    prompts_dir: Path,
    model: str,
    timeout_s: float,
    skip_existing: bool,
) -> str:
    """Process one ALT job. Returns 'ok', 'skipped', or 'failed'."""
    out_path = output_path_for(job.stem, output_dir)

    if skip_existing:
        existing = load_existing_output(out_path)
        if existing and existing.get("status") == "ok":
            return "skipped"

    alt_text = job.alt_text
    if source_kind == "pipeline":
        try:
            alt_text = load_final_alt_text(Path(job.source))
        except (ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
            record = build_record(
                job=job,
                source_kind=source_kind,
                model=model,
                result=None,
                processed_at=utc_now_iso(),
                error=str(exc),
            )
            save_output(out_path, record)
            print(f"  FAILED (input): {exc}")
            return "failed"
        job = AltJob(
            stem=job.stem,
            alt_text=alt_text,
            source=job.source,
            metadata=job.metadata,
        )

    result = run_stage_openai(
        client,
        stage=CLAIM_EXTRACTION_STAGE,
        tokens={"ALT_TEXT": alt_text},
        prompts_dir=prompts_dir,
        image_url="",
        single_model=model,
        timeout_s=timeout_s,
    )

    record = build_record(
        job=job,
        source_kind=source_kind,
        model=model,
        result=result,
        processed_at=utc_now_iso(),
    )
    save_output(out_path, record)

    if result.status == "ok" and isinstance(result.parsed, dict):
        claims = result.parsed.get("claims", [])
        print(f"  {len(claims)} claims — {result.duration_ms}ms — saved {out_path.name}")
        return "ok"

    print(f"  FAILED ({result.status}): {result.error}")
    return "failed"


def run(args: argparse.Namespace) -> int:
    configure_stdout()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.", file=sys.stderr)
        return 1

    root = project_root()
    output_dir = resolve_path(args.output_dir or default_output_dir(args.source))
    prompts_dir = root / PROMPTS_DIR
    client = OpenAI(api_key=api_key)

    try:
        jobs = resolve_jobs(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    title = "ALT-to-List Batch (ASSEST24)" if args.source == "asset24" else "ALT-to-List Batch (pipeline final ALT)"
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Source:   {args.source}")
    print(f"Model:    {args.model}")
    print(f"Output:   {output_dir}")
    print(f"Items:    {len(jobs)}")
    print()

    counts = {"ok": 0, "skipped": 0, "failed": 0}

    for i, job in enumerate(jobs, start=1):
        label = job.stem
        if args.source == "asset24" and "excel_row" in job.metadata:
            label = f"{job.stem} (row {job.metadata['excel_row']})"
        print(f"[{i}/{len(jobs)}] {label}")

        outcome = process_one(
            client=client,
            job=job,
            source_kind=args.source,
            output_dir=output_dir,
            prompts_dir=prompts_dir,
            model=args.model,
            timeout_s=float(args.timeout),
            skip_existing=args.skip_existing,
        )
        counts[outcome] += 1

        if i < len(jobs) and outcome != "skipped":
            time.sleep(args.delay)

    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"  Succeeded: {counts['ok']}")
    print(f"  Skipped:   {counts['skipped']}")
    print(f"  Failed:    {counts['failed']}")

    return 1 if counts["failed"] > 0 else 0


if __name__ == "__main__":
    cli_args = parse_args()
    if not cli_args.all and not cli_args.pipeline and cli_args.index is None:
        cli_args.all = True
    sys.exit(run(cli_args))
