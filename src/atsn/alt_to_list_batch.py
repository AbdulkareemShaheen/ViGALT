"""Backward-compatible shim for ``python -m atsn.alt_to_list_batch``."""

from atsn.evaluation.alt_to_list import main

if __name__ == "__main__":
    raise SystemExit(main())
