"""Backward-compatible shim for ``python -m atsn.combine_claims_relevancy``."""

from atsn.cli.combine_claims import main

if __name__ == "__main__":
    raise SystemExit(main())
