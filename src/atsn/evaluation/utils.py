"""Shared helpers for post-pipeline evaluator batch scripts."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..pipeline.utils import build_product_dom

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")

RELEVANT_CODES = frozenset({"R", "S"})
IRRELEVANT_CODES = frozenset({"V"})


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def summarize_results(results: list[dict]) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        algorithm = result.get("algorithm", "unknown")
        status = result.get("status", "error")
        bucket = summary.setdefault(algorithm, {"ok": 0, "failed": 0, "skipped": 0})
        if status == "ok":
            bucket["ok"] += 1
        elif status == "skipped":
            bucket["skipped"] += 1
        else:
            bucket["failed"] += 1
    return summary


def image_path_for_stem(product_stem: str, images_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        path = images_dir / f"{product_stem}{ext}"
        if path.exists() and path.stat().st_size > 0:
            return path.resolve()
    return None


def image_url_from_product_json(product_stem: str, products_dir: Path) -> str:
    product_file = products_dir / f"{product_stem}.json"
    if not product_file.exists():
        raise FileNotFoundError(f"Product JSON not found: {product_file}")

    data = json.loads(product_file.read_text(encoding="utf-8"))
    image_url = str(data.get("main_image") or "").strip()
    if not image_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid main_image URL in {product_file}")
    return image_url


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_image_reference(
    product_stem: str,
    *,
    images_dir: Path,
    products_dir: Path,
) -> tuple[str, str]:
    local_path = image_path_for_stem(product_stem, images_dir)
    if local_path is not None:
        return image_to_data_url(local_path), str(local_path)

    remote_url = image_url_from_product_json(product_stem, products_dir)
    return remote_url, remote_url


def load_product_dom(
    product_stem: str,
    products_dir: Path,
    *,
    product_file: str | Path | None = None,
) -> str:
    candidates: list[Path] = []
    if product_file:
        candidates.append(Path(product_file))
    candidates.append(products_dir / f"{product_stem}.json")

    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        return build_product_dom(data)

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Product JSON not found for stem {product_stem!r}. Checked: {searched}"
    )


def relevancy_job_key(algorithm: str, product_stem: str) -> str:
    return f"{algorithm}:{product_stem}"


def load_relevancy_results(
    path: Path,
    *,
    algorithm: str | None = None,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Relevancy evaluations not found: {path}")

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


def claims_with_relevance_from_relevancy(relevancy_result: dict) -> list[dict]:
    evaluation = relevancy_result["evaluation"]
    claims = evaluation.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Relevancy evaluation is missing claims.")

    input_claims: list[dict] = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Relevancy claim entry is not an object.")
        claim_id = claim.get("claim_id")
        claim_text = claim.get("claim")
        relevance_code = claim.get("label")
        if not claim_id or not claim_text or not relevance_code:
            raise ValueError("Relevancy claim is missing claim_id, claim, or label.")
        input_claims.append(
            {
                "claim_id": claim_id,
                "claim": claim_text,
                "relevance_code": relevance_code,
            }
        )
    return input_claims


def alt_text_from_relevancy(relevancy_result: dict) -> str:
    algorithm = relevancy_result.get("algorithm")
    if algorithm == "our":
        alt_text = relevancy_result.get("final_alt_text")
    elif algorithm == "asset24":
        alt_text = relevancy_result.get("asset24_alt_text")
    else:
        alt_text = None

    if not alt_text or not str(alt_text).strip():
        raise ValueError(
            f"Missing ALT text for algorithm {algorithm!r} in relevancy result "
            f"({relevancy_result.get('product_stem')})."
        )
    return str(alt_text).strip()


def validate_redundancy_output(llm_result: dict, input_claims: list[dict]) -> None:
    output_claims = llm_result.get("claims", [])
    if len(output_claims) != len(input_claims):
        raise ValueError(
            f"LLM returned {len(output_claims)} claims, expected {len(input_claims)}."
        )

    expected = {item["claim_id"]: item for item in input_claims}
    seen: set[str] = set()

    for output_claim in output_claims:
        claim_id = output_claim["claim_id"]
        if claim_id not in expected:
            raise ValueError(f"Unexpected claim_id returned: {claim_id}")
        if claim_id in seen:
            raise ValueError(f"Duplicate claim_id returned: {claim_id}")
        seen.add(claim_id)

        original = expected[claim_id]
        if output_claim["claim"] != original["claim"]:
            raise ValueError(f"{claim_id}: LLM modified the original claim text.")
        if output_claim["relevance_code"] != original["relevance_code"]:
            raise ValueError(f"{claim_id}: LLM modified the relevance code.")


def calculate_redundancy_rates(claims: list[dict]) -> dict:
    total = len(claims)

    relevant_total = sum(1 for c in claims if c["relevance_code"] in RELEVANT_CODES)
    irrelevant_total = sum(1 for c in claims if c["relevance_code"] in IRRELEVANT_CODES)

    raw_novel = sum(1 for c in claims if c["label"] == "NOVEL")
    relevant_novel = sum(
        1
        for c in claims
        if c["label"] == "NOVEL" and c["relevance_code"] in RELEVANT_CODES
    )
    irrelevant_novel = sum(
        1
        for c in claims
        if c["label"] == "NOVEL" and c["relevance_code"] in IRRELEVANT_CODES
    )
    unavoidable = sum(1 for c in claims if c["label"] == "UNAVOIDABLE_REPETITION")
    avoidable = sum(1 for c in claims if c["label"] == "AVOIDABLE_REDUNDANCY")

    if total == 0:
        return {
            "total_claims": 0,
            "relevant_claims": 0,
            "irrelevant_claims": 0,
            "raw_novel_claims": 0,
            "relevant_novel_claims": 0,
            "irrelevant_novel_claims": 0,
            "unavoidable_repetition_claims": 0,
            "avoidable_redundancy_claims": 0,
            "novelty_rate": None,
            "raw_novelty_rate": None,
            "unavoidable_repetition_rate": None,
            "avoidable_redundancy_rate": None,
            "redundancy_rate": None,
        }

    return {
        "total_claims": total,
        "relevant_claims": relevant_total,
        "irrelevant_claims": irrelevant_total,
        "raw_novel_claims": raw_novel,
        "relevant_novel_claims": relevant_novel,
        "irrelevant_novel_claims": irrelevant_novel,
        "unavoidable_repetition_claims": unavoidable,
        "avoidable_redundancy_claims": avoidable,
        "novelty_rate": relevant_novel / total,
        "raw_novelty_rate": raw_novel / total,
        "unavoidable_repetition_rate": unavoidable / total,
        "avoidable_redundancy_rate": avoidable / total,
        "redundancy_rate": (unavoidable + avoidable) / total,
    }


def calculate_objectivity_score(claims: list[dict]) -> dict:
    total_claims = len(claims)
    objective_claims = sum(1 for c in claims if c.get("label") == "OBJECTIVE")
    subjective_claims = sum(1 for c in claims if c.get("label") == "SUBJECTIVE")

    if total_claims == 0:
        objectivity_score = None
    else:
        objectivity_score = objective_claims / total_claims

    return {
        "objective_claims": objective_claims,
        "subjective_claims": subjective_claims,
        "total_claims": total_claims,
        "objectivity_score": objectivity_score,
    }


def count_alt_words(text: str) -> int:
    words = re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", text, flags=re.UNICODE)
    return len(words)


def calculate_efficiency(relevant_novel_claims: int, word_count: int) -> float | None:
    if word_count == 0:
        return None
    return (relevant_novel_claims / word_count) * 100
