"""OpenAI model and sampling configuration for the pipeline backend."""

from __future__ import annotations

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_EVALUATION_MODEL = "gpt-5.6-terra"

OPENAI_MODEL_MAP: dict[str, str] = {
    "classification": DEFAULT_OPENAI_MODEL,
    "generation": DEFAULT_OPENAI_MODEL,
    "claim_extraction": DEFAULT_OPENAI_MODEL,
    "accuracy_validation": DEFAULT_OPENAI_MODEL,
    "completeness_validation": DEFAULT_OPENAI_MODEL,
    "redundancy_validation": DEFAULT_OPENAI_MODEL,
    "fusion": DEFAULT_OPENAI_MODEL,
    "relevancy_evaluation": DEFAULT_EVALUATION_MODEL,
    "redundancy_evaluation": DEFAULT_EVALUATION_MODEL,
    "objectivity_evaluation": DEFAULT_EVALUATION_MODEL,
}

TEMPERATURE_MAP: dict[str, float] = {
    "generation": 0.3,
}

DEFAULT_TEMPERATURE = 0.0
FIXED_SEED = 42
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5.6")
DEFAULT_REASONING_EFFORT = "medium"


def resolve_openai_model(stage_name: str, single_model: str | None = None) -> str:
    if single_model:
        return single_model
    try:
        return OPENAI_MODEL_MAP[stage_name]
    except KeyError as exc:
        raise ValueError(f"No OpenAI model configured for stage: {stage_name}") from exc


def is_reasoning_model(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    return any(normalized.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)


def build_sampling_params(model: str, stage_name: str) -> dict:
    """Build temperature/seed or reasoning_effort params for a stage."""
    if is_reasoning_model(model):
        return {"reasoning_effort": DEFAULT_REASONING_EFFORT}
    temperature = TEMPERATURE_MAP.get(stage_name, DEFAULT_TEMPERATURE)
    return {"temperature": temperature, "seed": FIXED_SEED}


def strip_sampling_params(params: dict) -> dict:
    """Remove temperature and seed (e.g. when model rejects them)."""
    return {k: v for k, v in params.items() if k not in ("temperature", "seed")}
