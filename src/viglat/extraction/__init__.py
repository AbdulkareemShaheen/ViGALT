"""Product metadata extraction from Amazon pages."""

from .amazon import (
    AmazonExtractionError,
    extract_asin,
    extract_from_html,
    extract_from_url,
)

__all__ = [
    "AmazonExtractionError",
    "extract_asin",
    "extract_from_html",
    "extract_from_url",
]
