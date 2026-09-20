"""Backward-compatible shim for ``python -m atsn.relevancy_evaluator_batch``."""

from atsn.evaluation.relevancy import main

if __name__ == "__main__":
    raise SystemExit(main())
