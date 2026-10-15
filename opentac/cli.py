"""The `opentac` command. Subcommands are added as each part of the package lands."""

import argparse

from opentac import __version__


def _cad(args) -> int:
    from opentac.cad import launcher

    launcher.main(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opentac", description=__doc__)
    parser.add_argument("--version", action="version", version=f"opentac {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    cad = sub.add_parser("cad", help="design a sensor: start the CAD tool and open it in the browser")
    cad.add_argument("--host", default="127.0.0.1", help="address to serve on (default: this computer only)")
    cad.add_argument("--port", type=int, default=None, help="port (default: first free one from 8000)")
    cad.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    cad.set_defaults(func=_cad)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
