"""Backward-compatible shim for ``python -m viglat.redundancy_evaluator_batch``."""

from viglat.evaluation.redundancy import main

if __name__ == "__main__":
    raise SystemExit(main())
