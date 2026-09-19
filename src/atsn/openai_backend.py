"""OpenAI Chat Completions API backend."""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from .gemini_browser import is_noise_response
from .openai_config import (
    build_sampling_params,
    resolve_openai_model,
    strip_sampling_params,
)
from .openai_schemas import schema_for_stage
from .pipeline_types import StageConfig, StageResult
from .pipeline_utils import (
    build_user_message,
    load_prompt_template,
    split_prompt,
)

BACKEND_NAME = "openai_api"
MAX_RETRIES = 3
BACKOFF_BASE_S = 2.0


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is not None and int(status) >= 500:
        return True
    return False


def _temperature_rejected(exc: Exception) -> bool:
    message = str(exc).lower()
    return "temperature" in message or "unsupported" in message and "parameter" in message


def _build_messages(
    *,
    prompts_dir: Path,
    prompt_file: str,
    tokens: dict,
    image_url: str | None,
    uses_image: bool,
) -> list[dict]:
    template = load_prompt_template(prompts_dir, prompt_file)
    system_text, user_template = split_prompt(template)
    user_text = build_user_message(user_template, tokens)

    content: list[dict] = [{"type": "text", "text": user_text}]
    if uses_image and image_url:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "high"},
            }
        )

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


def _parse_json_stage(raw: str, expected_stage: str | None) -> dict:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    if expected_stage is not None and parsed.get("stage") != expected_stage:
        raise ValueError(
            f"Expected stage {expected_stage!r}, got {parsed.get('stage')!r}"
        )
    return parsed


def _token_usage_dict(usage) -> dict | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _call_openai(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict],
    stage: StageConfig,
    timeout_s: float,
    sampling_params: dict,
) -> tuple[str, dict | None]:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "timeout": timeout_s,
        **sampling_params,
    }
    response_format = schema_for_stage(stage.name)
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        if _temperature_rejected(exc) and ("temperature" in sampling_params or "seed" in sampling_params):
            retry_params = strip_sampling_params(sampling_params)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout_s,
                **retry_params,
                **({"response_format": response_format} if response_format else {}),
            )
        else:
            raise

    choice = response.choices[0]
    content = choice.message.content or ""
    return content, _token_usage_dict(response.usage)


def run_stage_openai(
    client: OpenAI,
    *,
    stage: StageConfig,
    tokens: dict,
    prompts_dir: Path,
    image_url: str,
    single_model: str | None,
    timeout_s: float,
) -> StageResult:
    model = resolve_openai_model(stage.name, single_model)
    messages = _build_messages(
        prompts_dir=prompts_dir,
        prompt_file=stage.prompt_file,
        tokens=tokens,
        image_url=image_url,
        uses_image=stage.uses_image,
    )
    sampling_params = build_sampling_params(model, stage.name)
    started = time.time()
    raw = ""
    token_usage: dict | None = None
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                wait_s = BACKOFF_BASE_S ** attempt
                print(f"Retrying stage {stage.name} in {wait_s:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait_s)

            raw, token_usage = _call_openai(
                client,
                model=model,
                messages=messages,
                stage=stage,
                timeout_s=timeout_s,
                sampling_params=sampling_params,
            )

            if stage.output_type == "text":
                parsed: dict | str = raw.strip()
                if not isinstance(parsed, str) or is_noise_response(parsed, min_length=20):
                    raise ValueError(
                        f"Generator returned invalid ALT text ({len(parsed or '')} chars): "
                        f"{parsed!r}"
                    )
            else:
                parsed = _parse_json_stage(raw, stage.expected_stage)

            duration_ms = int((time.time() - started) * 1000)
            usage_str = ""
            if token_usage:
                usage_str = (
                    f" | tokens: {token_usage['prompt_tokens']} prompt, "
                    f"{token_usage['completion_tokens']} completion, "
                    f"{token_usage['total_tokens']} total"
                )
            print(f"Stage {stage.name} completed in {duration_ms}ms{usage_str}")

            return StageResult(
                model=model,
                prompt_file=stage.prompt_file,
                raw_response=raw,
                parsed=parsed,
                duration_ms=duration_ms,
                status="ok",
                backend=BACKEND_NAME,
                token_usage=token_usage,
            )

        except json.JSONDecodeError as exc:
            duration_ms = int((time.time() - started) * 1000)
            return StageResult(
                model=model,
                prompt_file=stage.prompt_file,
                raw_response=raw,
                parsed=None,
                duration_ms=duration_ms,
                status="parse_error",
                backend=BACKEND_NAME,
                error=str(exc),
                token_usage=token_usage,
            )
        except ValueError as exc:
            if stage.output_type == "json" and raw:
                duration_ms = int((time.time() - started) * 1000)
                return StageResult(
                    model=model,
                    prompt_file=stage.prompt_file,
                    raw_response=raw,
                    parsed=None,
                    duration_ms=duration_ms,
                    status="parse_error",
                    backend=BACKEND_NAME,
                    error=str(exc),
                    token_usage=token_usage,
                )
            last_error = exc
            if not _is_transient_error(exc):
                break
        except Exception as exc:
            last_error = exc
            if not _is_transient_error(exc):
                break

    duration_ms = int((time.time() - started) * 1000)
    return StageResult(
        model=model,
        prompt_file=stage.prompt_file,
        raw_response=raw,
        parsed=None,
        duration_ms=duration_ms,
        status="error",
        backend=BACKEND_NAME,
        error=str(last_error) if last_error else "Unknown error",
        token_usage=token_usage,
    )
