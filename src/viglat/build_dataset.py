"""Backward-compatible shim for ``python -m viglat.build_dataset``."""

from viglat.cli.build_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
