"""Gemini web UI backend for pipeline stages."""

from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import Page

from .gemini_browser import is_noise_response, run_stage, save_screenshot
from .pipeline_types import StageConfig, StageResult
from .pipeline_utils import (
    build_user_message,
    load_prompt_template,
    parse_json_response,
    parse_plain_text,
    split_prompt,
)

BACKEND_NAME = "gemini_ui"


def _build_prompt(prompts_dir: Path, prompt_file: str, tokens: dict) -> str:
    template = load_prompt_template(prompts_dir, prompt_file)
    system_text, user_template = split_prompt(template)
    user_text = build_user_message(user_template, tokens)
    return f"{system_text}\n\n{user_text}"


def run_stage_gemini(
    page: Page,
    *,
    stage: StageConfig,
    tokens: dict,
    prompts_dir: Path,
    image_path: Path | None,
    timeout_ms: int,
    output_dir: Path,
    product_stem: str,
) -> StageResult:
    prompt = _build_prompt(prompts_dir, stage.prompt_file, tokens)
    model_key = stage.model
    started = time.time()
    raw = ""

    try:
        raw = run_stage(
            page,
            model_key=model_key,
            prompt=prompt,
            image_path=image_path,
            timeout_ms=timeout_ms,
        )

        screenshot_name = f"{product_stem}_{stage.name}.png"
        save_screenshot(page, output_dir, screenshot_name)

        if stage.output_type == "text":
            parsed: dict | str = parse_plain_text(raw)
            if not isinstance(parsed, str) or is_noise_response(parsed, min_length=20):
                raise ValueError(
                    f"Generator returned invalid ALT text ({len(parsed or '')} chars): "
                    f"{parsed!r}"
                )
        else:
            parsed = parse_json_response(raw, stage.expected_stage)

        duration_ms = int((time.time() - started) * 1000)
        print(f"Stage {stage.name} completed in {duration_ms}ms")

        return StageResult(
            model=model_key,
            prompt_file=stage.prompt_file,
            raw_response=raw,
            parsed=parsed,
            duration_ms=duration_ms,
            status="ok",
            backend=BACKEND_NAME,
        )

    except ValueError as exc:
        duration_ms = int((time.time() - started) * 1000)
        if raw:
            return StageResult(
                model=model_key,
                prompt_file=stage.prompt_file,
                raw_response=raw,
                parsed=None,
                duration_ms=duration_ms,
                status="parse_error",
                backend=BACKEND_NAME,
                error=str(exc),
            )
        return StageResult(
            model=model_key,
            prompt_file=stage.prompt_file,
            raw_response=raw,
            parsed=None,
            duration_ms=duration_ms,
            status="error",
            backend=BACKEND_NAME,
            error=str(exc),
        )

    except Exception as exc:
        duration_ms = int((time.time() - started) * 1000)
        return StageResult(
            model=model_key,
            prompt_file=stage.prompt_file,
            raw_response=raw,
            parsed=None,
            duration_ms=duration_ms,
            status="error",
            backend=BACKEND_NAME,
            error=str(exc),
        )
