"""Backward-compatible shim for ``python -m viglat.run_from_url``."""

from viglat.cli.run_from_url import main

if __name__ == "__main__":
    raise SystemExit(main())
