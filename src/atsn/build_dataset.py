"""Backward-compatible shim for ``python -m atsn.build_dataset``."""

from atsn.cli.build_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
