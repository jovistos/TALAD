"""Allow executing the package directly via `python3 -m talad`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
