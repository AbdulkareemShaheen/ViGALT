"""ViGALT: Visual Gap-Aware ALT Text Generation pipeline."""

from .types import StageConfig, StageResult
from .utils import ProductData, load_product

__all__ = [
    "ProductData",
    "StageConfig",
    "StageResult",
    "load_product",
]
