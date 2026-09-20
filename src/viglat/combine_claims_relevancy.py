"""Backward-compatible shim for ``python -m viglat.combine_claims_relevancy``."""

from viglat.cli.combine_claims import main

if __name__ == "__main__":
    raise SystemExit(main())
