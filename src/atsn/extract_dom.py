#!/usr/bin/env python3
"""
Extract product DOM metadata from an Amazon product page URL.

Writes JSON matching data/products/*.json and evaluation_dataset/*/dom.json.

Usage:
  python -m atsn.extract_dom --url "https://www.amazon.com/dp/B0..."
  python -m atsn.extract_dom --url "https://www.amazon.com/dp/B0..." --output-dir output/runs/B0...
  python -m atsn.extract_dom --html saved_page.html --output dom.json
  python -m atsn.extract_dom --url "https://www.amazon.com/dp/B0..." --stdout
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .amazon_extractor import (
    AmazonExtractionError,
    extract_asin,
    extract_from_html,
    extract_from_url,
)
from .pipeline_utils import download_image, image_filename_from_url, resolve_path


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Amazon product metadata into dom.json format."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--url",
        help="Amazon product page URL (must contain /dp/ASIN or /gp/product/ASIN).",
    )
    source.add_argument(
        "--html",
        help="Path to a locally saved Amazon product HTML page.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path. Ignored when --output-dir is set.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write {ASIN}.json and downloaded image into this directory.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds when fetching --url (default: 30).",
    )
    parser.add_argument(
        "--download-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download main_image next to the JSON file (default: true).",
    )
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace, record: dict) -> Path | None:
    if args.stdout:
        return None

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        asin = None
        if args.url:
            asin = extract_asin(args.url)
        if not asin:
            asin = output_dir.name or "product"
        return output_dir / f"{asin}.json"

    output_path = Path(args.output or "dom.json")
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    return output_path


def save_product_image(record: dict, output_path: Path) -> Path | None:
    main_image = record.get("main_image")
    if not main_image:
        return None

    url_name = image_filename_from_url(str(main_image))
    suffix = Path(url_name).suffix or ".jpg"
    image_path = output_path.parent / f"{output_path.stem}{suffix}"

    downloaded = download_image(str(main_image), output_path.parent)
    shutil.copy2(downloaded, image_path)
    return image_path.resolve()


def run(args: argparse.Namespace) -> int:
    configure_stdout()

    try:
        if args.html:
            html_path = resolve_path(args.html)
            html = html_path.read_text(encoding="utf-8", errors="replace")
            record = extract_from_html(html)
        else:
            record = extract_from_url(args.url, timeout_s=float(args.timeout))
    except AmazonExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not record.get("feature_bullets"):
        print(
            "Warning: no feature bullets extracted; page layout may differ or be incomplete.",
            file=sys.stderr,
        )
    if not record.get("product_details"):
        print(
            "Warning: no product details extracted; page layout may differ or be incomplete.",
            file=sys.stderr,
        )

    payload = json.dumps(record, indent=2, ensure_ascii=False) + "\n"

    if args.stdout:
        sys.stdout.write(payload)
        return 0

    output_path = resolve_output_path(args, record)
    if output_path is None:
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(f"Saved product DOM to: {output_path.resolve()}")

    if args.download_image:
        try:
            image_path = save_product_image(record, output_path)
            if image_path is not None:
                print(f"Saved product image to: {image_path}")
        except Exception as exc:
            print(f"Warning: could not download main_image: {exc}", file=sys.stderr)

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
