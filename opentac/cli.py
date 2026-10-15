"""The `opentac` command. Subcommands are added as each part of the package lands."""

import argparse

from opentac import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opentac", description=__doc__)
    parser.add_argument("--version", action="version", version=f"opentac {__version__}")
    parser.add_subparsers(dest="command", metavar="COMMAND")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
