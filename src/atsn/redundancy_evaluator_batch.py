"""Backward-compatible shim for ``python -m atsn.redundancy_evaluator_batch``."""

from atsn.evaluation.redundancy import main

if __name__ == "__main__":
    raise SystemExit(main())
