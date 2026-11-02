"""Explicitly provision one local rg executable for offline clean skill loading."""
import argparse
import json
from pathlib import Path

from ..core import runtime_tools


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rg", required=True, type=Path, help="Absolute path to an approved ripgrep executable")
    args = parser.parse_args(argv)
    print(json.dumps(runtime_tools.install(args.rg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
