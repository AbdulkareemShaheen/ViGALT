"""Backward-compatible shim for ``python -m viglat.extract_dom``."""

from viglat.cli.extract_dom import main

if __name__ == "__main__":
    raise SystemExit(main())
