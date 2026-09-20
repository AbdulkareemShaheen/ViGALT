"""Backward-compatible shim for ``python -m atsn.extract_dom``."""

from atsn.cli.extract_dom import main

if __name__ == "__main__":
    raise SystemExit(main())
