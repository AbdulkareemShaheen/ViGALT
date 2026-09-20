"""Backward-compatible shim for ``python -m viglat.efficiency_evaluator_batch``."""

from viglat.evaluation.efficiency import main

if __name__ == "__main__":
    raise SystemExit(main())
