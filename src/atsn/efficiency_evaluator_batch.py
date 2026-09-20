"""Backward-compatible shim for ``python -m atsn.efficiency_evaluator_batch``."""

from atsn.evaluation.efficiency import main

if __name__ == "__main__":
    raise SystemExit(main())
