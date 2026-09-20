"""Numbered run folder layout for ``run_from_url`` outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import project_root

RUNS_DIR = "output/runs"

DOM_FILENAME = "dom.json"
IMAGE_BASENAME = "image"
META_FILENAME = "meta.json"
PIPELINE_FILENAME = "pipeline.json"
CLAIMS_FILENAME = "claims.json"
RELEVANCY_FILENAME = "relevancy.json"
REDUNDANCY_FILENAME = "redundancy.json"
OBJECTIVITY_FILENAME = "objectivity.json"
EFFICIENCY_FILENAME = "efficiency.json"


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    work_dir: Path

    @property
    def dom_json(self) -> Path:
        return self.work_dir / DOM_FILENAME

    @property
    def meta_json(self) -> Path:
        return self.work_dir / META_FILENAME

    @property
    def pipeline_json(self) -> Path:
        return self.work_dir / PIPELINE_FILENAME

    @property
    def claims_json(self) -> Path:
        return self.work_dir / CLAIMS_FILENAME

    @property
    def relevancy_json(self) -> Path:
        return self.work_dir / RELEVANCY_FILENAME

    @property
    def redundancy_json(self) -> Path:
        return self.work_dir / REDUNDANCY_FILENAME

    @property
    def objectivity_json(self) -> Path:
        return self.work_dir / OBJECTIVITY_FILENAME

    @property
    def efficiency_json(self) -> Path:
        return self.work_dir / EFFICIENCY_FILENAME

    def image_path(self, suffix: str = ".jpg") -> Path:
        return self.work_dir / f"{IMAGE_BASENAME}{suffix}"

    def resolve_image_path(self) -> Path | None:
        from .pipeline.utils import find_local_product_image

        return find_local_product_image(self.work_dir)


def runs_root() -> Path:
    return project_root() / RUNS_DIR


def _is_numeric_run_id(name: str) -> bool:
    return name.isdigit() and int(name) > 0


def next_run_id(runs_dir: Path | None = None) -> int:
    root = runs_dir or runs_root()
    if not root.exists():
        return 1
    ids = [int(path.name) for path in root.iterdir() if path.is_dir() and _is_numeric_run_id(path.name)]
    return max(ids, default=0) + 1


def allocate_run_dir(
    *,
    work_dir: Path | None = None,
    run_id: int | None = None,
) -> RunPaths:
    if work_dir is not None:
        resolved = work_dir.resolve()
        return RunPaths(run_id=resolved.name, work_dir=resolved)

    root = runs_root()
    if run_id is not None:
        if run_id <= 0:
            raise ValueError(f"run_id must be a positive integer, got {run_id}")
        assigned = str(run_id)
    else:
        assigned = str(next_run_id(root))

    return RunPaths(run_id=assigned, work_dir=(root / assigned).resolve())
