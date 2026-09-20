"""Shared pipeline types used by orchestrator and backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class StageConfig:
    name: str
    prompt_file: str
    model: str
    uses_image: bool
    output_type: Literal["json", "text"]
    expected_stage: str | None = None


@dataclass
class StageResult:
    model: str
    prompt_file: str
    raw_response: str
    parsed: dict | str | None
    duration_ms: int
    status: str
    backend: str
    error: str | None = None
    token_usage: dict | None = None
