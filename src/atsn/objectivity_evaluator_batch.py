"""Backward-compatible shim for ``python -m atsn.objectivity_evaluator_batch``."""

from atsn.evaluation.objectivity import main

if __name__ == "__main__":
    raise SystemExit(main())
