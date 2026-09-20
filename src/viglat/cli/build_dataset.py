#!/usr/bin/env python3
"""
Build a consolidated per-image evaluation dataset from raw Dataset/ folders.

Reads Dataset/N/ (image, DOM, alt texts, claim relevancy, evaluation JSONs)
and writes evaluation_dataset/N/ with alt_text_and_claims.txt (alt text +
plain claims, no relevancy labels), evaluation.json, plus a top-level
summary.json with averaged metrics.

Renames the AAAA method to ViGALT everywhere.

Usage:
  python -m viglat.build_dataset
  python -m viglat.build_dataset --source Dataset --output evaluation_dataset
  python -m viglat.build_dataset --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ..evaluation.utils import IMAGE_EXTENSIONS, RELEVANT_CODES
from ..paths import resolve_path

DEFAULT_SOURCE = "Dataset"
DEFAULT_OUTPUT = "evaluation_dataset"

SOURCE_METHODS: dict[str, str] = {
    "AAAA": "ViGALT",
    "ASSETS24": "ASSETS24",
}

SOURCE_ALT_FILES: dict[str, str] = {
    "AAAA": "AAAA_{n}.txt",
    "ASSETS24": "ASSETS24_{n}.txt",
}

SOURCE_CLAIM_FILES: dict[str, str] = {
    "AAAA": "CLAIM_RELEVANCY_AAAA_{n}.txt",
    "ASSETS24": "CLAIM_RELEVANCY_ASSETS_{n}.txt",
}

SUMMARY_METRICS = (
    "relevancy_rate",
    "objectivity_score",
    "efficiency_per_100_words",
    "novelty_rate",
    "raw_novelty_rate",
    "avoidable_redundancy_rate",
    "unavoidable_repetition_rate",
    "redundancy_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build consolidated evaluation_dataset/ from raw Dataset/ folders."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Source directory with numbered subfolders (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory contents.",
    )
    return parser.parse_args()


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def discover_folders(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    folders = [path for path in source.iterdir() if path.is_dir() and path.name.isdigit()]
    if not folders:
        raise FileNotFoundError(f"No numbered subfolders found in: {source}")

    return sorted(folders, key=lambda path: int(path.name))


def find_image(folder: Path) -> Path:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    raise FileNotFoundError(f"No image file found in {folder}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def parse_claims(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")

    claims: list[dict] = []
    for index, raw_line in enumerate(read_text(path).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "|" not in line:
            raise ValueError(f"Invalid claim line in {path}: {line!r}")

        claim_text, label = line.rsplit("|", 1)
        claim_text = claim_text.strip()
        label = label.strip()
        if not claim_text or not label:
            raise ValueError(f"Invalid claim line in {path}: {line!r}")

        claims.append(
            {
                "claim_id": f"C{index}",
                "claim": claim_text,
                "label": label,
            }
        )
    return claims


def relevancy_block(claims: list[dict]) -> dict:
    total = len(claims)
    relevant = sum(1 for claim in claims if claim["label"] in RELEVANT_CODES)
    supplementary = sum(1 for claim in claims if claim["label"] == "S")
    visual_only = sum(1 for claim in claims if claim["label"] == "V")

    return {
        "total_claims": total,
        "relevant_claims": relevant,
        "supplementary_claims": supplementary,
        "visual_only_claims": visual_only,
        "relevancy_rate": (relevant / total) if total else None,
        "claims": claims,
    }


def format_alt_text_and_claims(
    alt_texts: dict[str, str],
    claims_by_method: dict[str, list[dict]],
) -> str:
    sections: list[str] = []
    for method in SOURCE_METHODS.values():
        alt_text = alt_texts[method]
        claim_lines = [claim["claim"] for claim in claims_by_method[method]]
        body = alt_text
        if claim_lines:
            body = f"{alt_text}\n\n" + "\n".join(claim_lines)
        sections.append(f"=== {method} ===\n{body}")
    return "\n\n".join(sections) + "\n"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_method_eval(
    *,
    source_key: str,
    output_key: str,
    alt_text: str,
    claims: list[dict],
    efficiency_data: dict,
    objectivity_data: dict,
    redundancy_data: dict,
) -> dict:
    relevancy = relevancy_block(claims)
    redundancy_metrics = redundancy_data.get("metrics", {})
    objectivity_claims = objectivity_data.get("llm_output", {}).get("claims", [])
    redundancy_claims = redundancy_data.get("llm_output", {}).get("claims", [])

    return {
        "alt_text": alt_text,
        "relevancy": relevancy,
        "redundancy": {
            "metrics": redundancy_metrics,
            "input_claims": redundancy_data.get("input_claims", []),
            "claims": redundancy_claims,
        },
        "objectivity": {
            "objectivity_score": objectivity_data.get("objectivity_score"),
            "objective_claims": objectivity_data.get("objective_claims"),
            "subjective_claims": objectivity_data.get("subjective_claims"),
            "total_claims": objectivity_data.get("total_claims"),
            "claims": objectivity_claims,
        },
        "efficiency": {
            "word_count": efficiency_data.get("word_count"),
            "relevant_novel_claims": efficiency_data.get("relevant_novel_claims"),
            "efficiency_per_100_words": efficiency_data.get("efficiency_per_100_words"),
        },
        "_source_method_key": source_key,
        "_output_method_key": output_key,
    }


def build_folder_evaluation(
    *,
    folder_number: int,
    image_file: str,
    alt_texts: dict[str, str],
    claims_by_method: dict[str, list[dict]],
    efficiency_json: dict,
    objectivity_json: dict,
    redundancy_json: dict,
) -> dict:
    methods: dict[str, dict] = {}

    for source_key, output_key in SOURCE_METHODS.items():
        methods[output_key] = build_method_eval(
            source_key=source_key,
            output_key=output_key,
            alt_text=alt_texts[output_key],
            claims=claims_by_method[output_key],
            efficiency_data=efficiency_json["results"][source_key],
            objectivity_data=objectivity_json[source_key],
            redundancy_data=redundancy_json[source_key],
        )
        methods[output_key].pop("_source_method_key", None)
        methods[output_key].pop("_output_method_key", None)

    return {
        "folder_number": folder_number,
        "image_file": image_file,
        "methods": methods,
    }


def metric_value(method_eval: dict, metric: str) -> float | None:
    if metric == "relevancy_rate":
        return method_eval["relevancy"].get("relevancy_rate")
    if metric == "objectivity_score":
        return method_eval["objectivity"].get("objectivity_score")
    if metric == "efficiency_per_100_words":
        return method_eval["efficiency"].get("efficiency_per_100_words")
    if metric in {
        "novelty_rate",
        "raw_novelty_rate",
        "avoidable_redundancy_rate",
        "unavoidable_repetition_rate",
        "redundancy_rate",
    }:
        return method_eval["redundancy"]["metrics"].get(metric)
    raise ValueError(f"Unknown summary metric: {metric}")


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_summary(all_evaluations: list[dict]) -> dict:
    summary: dict = {"folders_counted": len(all_evaluations), "methods": {}}

    for method in SOURCE_METHODS.values():
        method_values: dict[str, list[float]] = {metric: [] for metric in SUMMARY_METRICS}
        total_claims = 0

        for evaluation in all_evaluations:
            method_eval = evaluation["methods"][method]
            total_claims += method_eval["relevancy"]["total_claims"]
            for metric in SUMMARY_METRICS:
                value = metric_value(method_eval, metric)
                if value is not None:
                    method_values[metric].append(float(value))

        summary["methods"][method] = {
            "total_claims": total_claims,
            **{
                f"avg_{metric}": average(method_values[metric])
                for metric in SUMMARY_METRICS
            },
        }

    return summary


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_folder(source_folder: Path, output_folder: Path) -> dict:
    folder_number = int(source_folder.name)

    image_path = find_image(source_folder)
    dom_source = source_folder / f"DOM_JSON_{folder_number}.txt"
    efficiency_path = source_folder / f"efficiency_results_{folder_number}.json"
    objectivity_path = source_folder / f"objectivity_results_{folder_number}.json"
    redundancy_path = source_folder / f"redundancy_results_{folder_number}.json"

    alt_texts: dict[str, str] = {}
    claims_by_method: dict[str, list[dict]] = {}

    for source_key, output_key in SOURCE_METHODS.items():
        alt_path = source_folder / SOURCE_ALT_FILES[source_key].format(n=folder_number)
        claim_path = source_folder / SOURCE_CLAIM_FILES[source_key].format(n=folder_number)
        alt_texts[output_key] = read_text(alt_path)
        claims_by_method[output_key] = parse_claims(claim_path)

    efficiency_json = load_json(efficiency_path)
    objectivity_json = load_json(objectivity_path)
    redundancy_json = load_json(redundancy_path)

    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_folder / image_path.name)
    shutil.copy2(dom_source, output_folder / "dom.json")

    alt_text_and_claims_path = output_folder / "alt_text_and_claims.txt"
    evaluation_path = output_folder / "evaluation.json"

    alt_text_and_claims_path.write_text(
        format_alt_text_and_claims(alt_texts, claims_by_method),
        encoding="utf-8",
    )

    evaluation = build_folder_evaluation(
        folder_number=folder_number,
        image_file=image_path.name,
        alt_texts=alt_texts,
        claims_by_method=claims_by_method,
        efficiency_json=efficiency_json,
        objectivity_json=objectivity_json,
        redundancy_json=redundancy_json,
    )
    write_json(evaluation_path, evaluation)
    return evaluation


def prepare_output_dir(output_root: Path, force: bool) -> None:
    if output_root.exists() and force:
        for child in output_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    configure_stdout()

    source_root = resolve_path(args.source)
    output_root = resolve_path(args.output)
    prepare_output_dir(output_root, args.force)

    folders = discover_folders(source_root)
    evaluations: list[dict] = []

    print("=" * 60)
    print("Build Consolidated Evaluation Dataset")
    print("=" * 60)
    print(f"Source:  {source_root}")
    print(f"Output:  {output_root}")
    print(f"Folders: {len(folders)}")
    print()

    for source_folder in folders:
        folder_number = source_folder.name
        output_folder = output_root / folder_number
        evaluation = process_folder(source_folder, output_folder)
        evaluations.append(evaluation)
        print(f"[{folder_number}] wrote {output_folder.name}/")

    summary = build_summary(evaluations)
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Folders: {summary['folders_counted']}")
    for method, metrics in summary["methods"].items():
        print(f"  {method}:")
        print(f"    avg_relevancy_rate:          {metrics['avg_relevancy_rate']:.4f}")
        print(f"    avg_objectivity_score:       {metrics['avg_objectivity_score']:.4f}")
        print(f"    avg_efficiency_per_100_words:{metrics['avg_efficiency_per_100_words']:.4f}")
        print(f"    avg_novelty_rate:            {metrics['avg_novelty_rate']:.4f}")
        print(f"    avg_redundancy_rate:         {metrics['avg_redundancy_rate']:.4f}")
    print(f"Saved: {summary_path}")

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
