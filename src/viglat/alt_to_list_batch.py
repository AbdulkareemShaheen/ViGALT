"""Backward-compatible shim for ``python -m viglat.alt_to_list_batch``."""

from viglat.evaluation.alt_to_list import main

if __name__ == "__main__":
    raise SystemExit(main())
