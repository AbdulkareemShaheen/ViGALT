"""Extract product metadata from Amazon product pages into dom.json schema."""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from ..pipeline.utils import normalize_product_record, validate_product_record

AMAZON_HOST_PATTERN = re.compile(r"(^|\.)amazon\.(com|co\.uk|de|fr|it|es|ca|com\.au|in|nl|se|pl|com\.be|com\.mx|com\.br|co\.jp|ae|sa|sg|com\.tr)$")
ASIN_PATTERN = re.compile(r"(?:/dp/|/gp/product/|/gp/aw/d/|/product/)([A-Z0-9]{10})", re.I)
ASIN_QUERY_PATTERN = re.compile(r"(?:[?&]ASIN=|(?:[?&])asin=)([A-Z0-9]{10})", re.I)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class AmazonExtractionError(Exception):
    """Raised when Amazon product metadata cannot be extracted."""


def is_amazon_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return bool(AMAZON_HOST_PATTERN.search(host))


def extract_asin(url: str) -> str | None:
    for pattern in (ASIN_PATTERN, ASIN_QUERY_PATTERN):
        match = pattern.search(url)
        if match:
            return match.group(1).upper()
    return None


def normalize_amazon_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise AmazonExtractionError(f"Invalid URL: {url!r}")
    if not is_amazon_url(url):
        raise AmazonExtractionError(
            f"Not an Amazon product URL: {url!r}. Only Amazon domains are supported."
        )
    asin = extract_asin(url)
    if asin is None:
        raise AmazonExtractionError(
            f"Could not extract ASIN from URL: {url!r}. "
            "Use a link containing /dp/ASIN or /gp/product/ASIN."
        )
    return f"https://{parsed.netloc}/dp/{asin}"


def fetch_amazon_html(url: str, *, timeout_s: float = 30.0) -> str:
    canonical_url = normalize_amazon_url(url)
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = session.get(canonical_url, timeout=timeout_s)
            if response.status_code == 503 and attempt == 0:
                continue
            response.raise_for_status()
            html = response.text
            if len(html) < 500:
                raise AmazonExtractionError("Amazon returned an unexpectedly short response.")
            return html
        except requests.RequestException as exc:
            last_error = exc

    message = f"Failed to fetch Amazon page: {canonical_url}"
    if last_error is not None:
        raise AmazonExtractionError(message) from last_error
    raise AmazonExtractionError(message)


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _upgrade_image_url(url: str) -> str:
    cleaned = url.strip()
    if cleaned.startswith("//"):
        cleaned = f"https:{cleaned}"
    cleaned = re.sub(r"\._AC_[A-Z0-9_]+_\.", ".", cleaned)
    return cleaned


def _first_string(value: object) -> str | None:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        for item in value:
            text = _first_string(item)
            if text:
                return text
    if isinstance(value, dict):
        for key in ("@value", "name", "value"):
            if key in value:
                text = _first_string(value[key])
                if text:
                    return text
    return None


def _parse_json_ld(soup: BeautifulSoup) -> dict:
    result: dict = {}
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type")
            types = item_type if isinstance(item_type, list) else [item_type]
            if "Product" not in types:
                continue

            if not result.get("title"):
                result["title"] = _first_string(item.get("name"))

            if not result.get("brand"):
                brand = item.get("brand")
                if isinstance(brand, dict):
                    result["brand"] = _first_string(brand.get("name"))
                else:
                    result["brand"] = _first_string(brand)

            if not result.get("description"):
                result["description"] = _first_string(item.get("description"))

            if not result.get("main_image"):
                image = item.get("image")
                if isinstance(image, list):
                    result["main_image"] = _first_string(image[0]) if image else None
                else:
                    result["main_image"] = _first_string(image)

            feature_list = item.get("featureList") or item.get("features")
            if isinstance(feature_list, list):
                bullets = result.setdefault("feature_bullets", [])
                for feature in feature_list:
                    text = _first_string(feature)
                    if text and text not in bullets:
                        bullets.append(text)

    return result


def _text_from_selector(soup: BeautifulSoup, selector: str) -> str | None:
    element = soup.select_one(selector)
    if element is None:
        return None
    return _clean_text(element.get_text(" ", strip=True))


def _parse_feature_bullets(soup: BeautifulSoup) -> list[str]:
    bullets: list[str] = []
    for element in soup.select("#feature-bullets li span.a-list-item"):
        text = _clean_text(element.get_text(" ", strip=True))
        if text and text not in bullets:
            bullets.append(text)
    return bullets


def _parse_description(soup: BeautifulSoup) -> str | None:
    for selector in (
        "#productDescription p",
        "#productDescription",
        "#aplus_feature_div",
        "#aplus",
    ):
        text = _text_from_selector(soup, selector)
        if text:
            return text
    return None


def _append_detail_pair(details: dict[str, str], label: str | None, value: str | None) -> None:
    if label and value and label not in details:
        details[label] = value


def _parse_detail_tables(soup: BeautifulSoup) -> dict[str, str]:
    details: dict[str, str] = {}

    for row in soup.select(
        "#productDetails_detailBullets_sections1 tr, "
        "#productDetails_techSpec_section_1 tr, "
        "#productDetails_techSpec_section_2 tr"
    ):
        header = row.find("th")
        cell = row.find("td")
        if header and cell:
            _append_detail_pair(
                details,
                _clean_text(header.get_text(" ", strip=True)),
                _clean_text(cell.get_text(" ", strip=True)),
            )

    for row in soup.select("#detailBullets_feature_div li"):
        text = _clean_text(row.get_text(" ", strip=True))
        if not text:
            continue
        if ":" in text:
            label, value = text.split(":", 1)
            _append_detail_pair(details, _clean_text(label), _clean_text(value))

    for table in soup.select("table.prodDetTable"):
        for row in table.select("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                _append_detail_pair(
                    details,
                    _clean_text(cells[0].get_text(" ", strip=True)),
                    _clean_text(cells[1].get_text(" ", strip=True)),
                )

    return details


def _parse_main_image(soup: BeautifulSoup) -> str | None:
    for selector in ("#landingImage", "#imgTagWrapperId img", "#main-image"):
        element = soup.select_one(selector)
        if element is None:
            continue
        for attr in ("data-old-hires", "data-a-dynamic-image", "src"):
            raw = element.get(attr)
            if not raw:
                continue
            if attr == "data-a-dynamic-image":
                try:
                    images = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(images, dict) and images:
                    url = next(iter(images))
                    return _upgrade_image_url(url)
            else:
                return _upgrade_image_url(raw)
    return None


def _parse_main_image_alt(soup: BeautifulSoup, *, title: str | None) -> str | None:
    for selector in ("#landingImage", "#imgTagWrapperId img", "#altImages img"):
        element = soup.select_one(selector)
        if element is not None:
            alt = _clean_text(element.get("alt"))
            if alt and alt.lower() not in {"image", "product image"}:
                return alt
    return title


def extract_from_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    merged: dict = {}

    json_ld = _parse_json_ld(soup)
    merged.update(json_ld)

    if not merged.get("title"):
        merged["title"] = _text_from_selector(soup, "#productTitle")

    if not merged.get("brand"):
        merged["brand"] = _text_from_selector(soup, "#bylineInfo")

    if not merged.get("description"):
        merged["description"] = _parse_description(soup)

    if not merged.get("feature_bullets"):
        merged["feature_bullets"] = _parse_feature_bullets(soup)

    if not merged.get("product_details"):
        merged["product_details"] = _parse_detail_tables(soup)

    if not merged.get("main_image"):
        merged["main_image"] = _parse_main_image(soup)

    if not merged.get("main_image_alt"):
        merged["main_image_alt"] = _parse_main_image_alt(
            soup,
            title=merged.get("title"),
        )

    normalized = normalize_product_record(merged)
    validate_product_record(normalized)
    return normalized


def extract_from_url(url: str, *, timeout_s: float = 30.0) -> dict:
    html = fetch_amazon_html(url, timeout_s=timeout_s)
    return extract_from_html(html)
