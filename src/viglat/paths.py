"""Stable project-root resolution regardless of subpackage nesting."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent


def load_env() -> None:
    """Load ``.env`` from the project root into ``os.environ`` (if present)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass


load_env()


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()
