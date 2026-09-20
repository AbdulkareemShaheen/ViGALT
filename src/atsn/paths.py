"""Stable project-root resolution regardless of subpackage nesting."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()
