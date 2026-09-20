"""Product loading, prompt templating, and response parsing for the pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

DOWNLOADS_DIR = "downloads"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@dataclass
class ProductData:
    product_file: Path
    image_url: str
    image_path: Path
    surrounding_text: str
    product_dom: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def image_filename_from_url(image_url: str) -> str:
    path = urlparse(image_url).path
    name = Path(path).name
    if name and "." in name:
        return name
    return "downloaded_image.jpg"


def download_image(image_url: str, downloads_dir: Path) -> Path:
    downloads_dir.mkdir(parents=True, exist_ok=True)
    filename = image_filename_from_url(image_url)
    image_path = downloads_dir / filename

    if image_path.exists() and image_path.stat().st_size > 0:
        print(f"Using cached image: {image_path}")
        return image_path.resolve()

    print(f"Downloading image: {image_url}")
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_path.write_bytes(response.content)
    print(f"Saved image to: {image_path}")
    return image_path.resolve()


def build_surrounding_text(data: dict) -> str:
    parts: list[str] = []

    title = data.get("title")
    if title:
        parts.append(f"Title: {title}")

    brand = data.get("brand")
    if brand:
        parts.append(f"Brand: {brand}")

    description = data.get("description")
    if description:
        parts.append(f"Description: {description}")

    bullets = data.get("feature_bullets")
    if isinstance(bullets, list) and bullets:
        parts.append("Feature bullets:")
        parts.extend(f"- {item}" for item in bullets)

    details = data.get("product_details")
    if isinstance(details, dict) and details:
        parts.append("Product details:")
        for key, value in details.items():
            parts.append(f"- {key}: {value}")

    alt = data.get("main_image_alt")
    if alt:
        parts.append(f"Main image alt: {alt}")

    return "\n".join(parts)


def build_product_dom(data: dict) -> str:
    dom = {k: v for k, v in data.items() if k != "main_image"}
    return json.dumps(dom, indent=2, ensure_ascii=False)


def load_product(product_file: Path, downloads_dir: Path | None = None) -> ProductData:
    if not product_file.exists():
        raise FileNotFoundError(f"Product file not found: {product_file}")

    data = json.loads(product_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{product_file} must contain a JSON object.")

    image_url = data.get("main_image")
    if not image_url:
        raise ValueError(f"{product_file} is missing a 'main_image' field.")

    image_url = str(image_url).strip()
    if not image_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid main_image URL in {product_file}: {image_url!r}")

    if downloads_dir is None:
        downloads_dir = project_root() / DOWNLOADS_DIR

    image_path = download_image(image_url, downloads_dir)
    return ProductData(
        product_file=product_file.resolve(),
        image_url=image_url,
        image_path=image_path,
        surrounding_text=build_surrounding_text(data),
        product_dom=build_product_dom(data),
    )


def load_prompt_template(prompts_dir: Path, prompt_file: str) -> str:
    path = prompts_dir / prompt_file
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").rstrip()


def _token_value(value: str | dict | list) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def fill_prompt(template: str, tokens: dict[str, str | dict | list]) -> str:
    result = template
    for key, value in tokens.items():
        result = result.replace(f"{{{{{key}}}}}", _token_value(value))
    return result


INPUT_MARKER = "# INPUT"


def split_prompt(template: str) -> tuple[str, str]:
    """Split at the first '# INPUT' line. Returns (system_text, user_template)."""
    match = re.search(r"^# INPUT\s*$", template, re.MULTILINE)
    if not match:
        raise ValueError(f"Prompt template missing {INPUT_MARKER!r} marker.")
    system_text = template[: match.start()].rstrip()
    user_template = template[match.end() :].lstrip("\n")
    return system_text, user_template


def build_user_message(user_template: str, tokens: dict[str, str | dict | list]) -> str:
    """Fill token placeholders in the user section only."""
    return fill_prompt(user_template, tokens)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", stripped, re.DOTALL | re.I)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


RESPONSE_NOISE_PATTERNS = [
    re.compile(r"^opens in a new window\.?$", re.I),
    re.compile(r"^(copy|share|more|edit|retry)$", re.I),
    re.compile(r"^gemini$", re.I),
]


def is_noise_response(text: str, *, min_length: int = 15) -> bool:
    cleaned = text.strip()
    if len(cleaned) < min_length:
        return True
    for pattern in RESPONSE_NOISE_PATTERNS:
        if pattern.search(cleaned):
            return True
    return False


def _strip_gemini_prefix(text: str) -> str:
    return re.sub(r"^Gemini said\s*\n+", "", text.strip(), flags=re.I).strip()


def _repair_gemini_json(text: str) -> str:
    """Fix common Gemini JSON mistakes such as ""WORD" instead of \"WORD\"."""
    patterns = (
        r'("claim":\s*)""([^"]+)"',
        r'("original_claim":\s*)""([^"]+)"',
        r'("corrected_claim":\s*)""([^"]+)"',
        r'("text":\s*)""([^"]+)"',
    )
    repaired = text
    for pattern in patterns:
        repaired = re.sub(pattern, r'\1"\\"\2\\"', repaired)
    inline_quote_patterns = (
        (r' and "([^"]+)" ', r' and \\"\1\\" '),
        (r' display "([^"]+)" ', r' display \\"\1\\" '),
        (r' featuring "([^"]+)" ', r' featuring \\"\1\\" '),
        (r' with "([^"]+)" ', r' with \\"\1\\" '),
        (r' reads "([^"]+)" ', r' reads \\"\1\\" '),
        (r' says "([^"]+)" ', r' says \\"\1\\" '),
        (r' showing "([^"]+)" ', r' showing \\"\1\\" '),
    )
    for pattern, replacement in inline_quote_patterns:
        repaired = re.sub(pattern, replacement, repaired)
    return repaired


def parse_json_response(raw: str, expected_stage: str | None = None) -> dict:
    cleaned = _strip_gemini_prefix(raw)
    extracted = _extract_json_object(_strip_markdown_fences(cleaned))
    candidates = [
        cleaned,
        _strip_markdown_fences(cleaned),
        extracted,
        _repair_gemini_json(cleaned),
        _repair_gemini_json(_strip_markdown_fences(cleaned)),
        _repair_gemini_json(extracted),
    ]

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
            if expected_stage is not None and parsed.get("stage") != expected_stage:
                raise ValueError(
                    f"Expected stage {expected_stage!r}, got {parsed.get('stage')!r}"
                )
            return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    msg = f"Could not parse JSON response: {raw[:200]!r}"
    if last_error is not None:
        raise ValueError(msg) from last_error
    raise ValueError(msg)


def parse_plain_text(raw: str) -> str:
    text = _strip_gemini_prefix(raw)
    if text.startswith("```"):
        text = _strip_markdown_fences(text)
    return text.strip()


def list_product_files(products_dir: Path) -> list[Path]:
    if not products_dir.exists():
        raise FileNotFoundError(f"Products folder not found: {products_dir}")

    products = sorted(
        path.resolve()
        for path in products_dir.glob("*.json")
        if path.is_file()
    )
    if not products:
        raise FileNotFoundError(f"No product JSON files found in: {products_dir}")
    return products
