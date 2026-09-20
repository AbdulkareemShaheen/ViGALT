"""OpenAI API backend for pipeline and evaluator stages."""

from .config import DEFAULT_OPENAI_MODEL, resolve_openai_model

__all__ = ["DEFAULT_OPENAI_MODEL", "resolve_openai_model", "run_stage_openai"]


def __getattr__(name: str):
    if name == "run_stage_openai":
        from .openai import run_stage_openai

        return run_stage_openai
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
