"""Backward-compatible shim for ``python -m atsn.run_from_url``."""

from atsn.cli.run_from_url import main

if __name__ == "__main__":
    raise SystemExit(main())
