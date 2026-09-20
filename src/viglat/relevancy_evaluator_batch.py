"""Backward-compatible shim for ``python -m viglat.relevancy_evaluator_batch``."""

from viglat.evaluation.relevancy import main

if __name__ == "__main__":
    raise SystemExit(main())
