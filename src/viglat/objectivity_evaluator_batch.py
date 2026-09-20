"""Backward-compatible shim for ``python -m viglat.objectivity_evaluator_batch``."""

from viglat.evaluation.objectivity import main

if __name__ == "__main__":
    raise SystemExit(main())
