"""The `opentac` command. Run `opentac COMMAND --help` for each command's options."""

import argparse
import sys
import warnings

from opentac import __version__


def _cad(args) -> int:
    from opentac.cad import launcher

    launcher.main(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


# ----- readout boards -----------------------------------------------------------

def _settings(pairs):
    out = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set takes NAME=VALUE, got {pair!r}")
        out[name.strip()] = value.strip()
    return out


#: Open a board without changing anything on it.
READ_ONLY = dict(raster=False, mux_settle_us=None, imu=None)


def _session(args) -> dict:
    return dict(imu=getattr(args, "imu", False), settings=_settings(getattr(args, "set", None)))


def _open(args, all_nodes=False, session=None):
    """(sensors, bus) for the board or bus the arguments point at."""
    from opentac.device import Bus, Sensor
    from opentac.device.link import find_any

    session = _session(args) if session is None else session
    node = getattr(args, "node", None)
    if node is None and all_nodes and not args.port:
        kind, port = find_any()
        if kind == "bus":
            bus = Bus(port, **session)
            return list(bus), bus
        return [Sensor(port, **session)], None
    if node is None:
        return [Sensor(args.port, **session)], None
    bus = Bus(args.port, **session)
    if node == "pick":
        print("which board? its LED is lit red")
        s = bus.pick()
        if s is None:
            raise SystemExit("cancelled")
        return [s], bus
    if node == "all":
        return list(bus), bus
    if not node.isdigit():
        raise SystemExit(f"--node takes a number, `pick` or `all`, not {node!r}")
    return [bus[int(node)]], bus


def _close(sensors, bus):
    if bus is not None:
        bus.close()
    else:
        for s in sensors:
            s.close()


def _describe_skin(stored, indent="  "):
    shear = f"{stored['shear']} taxels grouped" if stored["shear"] else "none (pressure only)"
    return "\n".join(indent + line for line in (
        f"array       {stored['rows']} x {stored['cols']}, {stored['active']} taxels",
        f"pitch       {stored['pitch_x_um'] / 1000:.3f} x {stored['pitch_y_um'] / 1000:.3f} mm",
        f"shear       {shear}",
        f"skin id     {stored['config_id']:08X}  (crc {stored['crc']:08X}, {stored['bytes']} bytes)"))


def _setup(args) -> int:
    from opentac import manifest
    from opentac.device import board, build_config, cache_path, extract
    from opentac.device.config import describe_lines

    m = manifest.load(args.manifest, scan=True)
    if args.dry_run:
        if not args.board:
            raise SystemExit("--dry-run needs --board (readout60, node16 or readout32)")
        b = board(args.board)
        spec = extract(m.raw, b)
        blob = build_config(spec, m.config_id)
        for note in m.warnings + spec["warnings"]:
            print(f"warning: {note}")
        print(f"{m.name}: {spec['rows']} x {spec['cols']}, {len(spec['active'])} taxels on {b.name}")
        print(f"  rows    -> {describe_lines(spec['row_order'])}")
        print(f"  columns -> {describe_lines(spec['col_order'])}")
        print(f"  config  {len(blob)} bytes, skin id {m.config_id:08X}; nothing sent")
        if args.out:
            with open(args.out, "wb") as fh:
                fh.write(blob)
            print(f"  written to {args.out}")
        return 0
    sensors, bus = _open(args, session=READ_ONLY)
    try:
        s = sensors[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stored = s.setup(m)
        for w in caught:
            print(f"warning: {w.message}")
        where = f"node {s.node}" if s.node is not None else s.link.port
        print(f"{s.board_name} on {where} now holds {m.name}:")
        print(_describe_skin(stored))
        print(f"  manifest kept at {cache_path(stored['config_id'])}")
    finally:
        _close(sensors, bus)
    return 0


def _info(args) -> int:
    from opentac.device import Bus
    from opentac.device.link import find_any

    if args.node is None and not args.port:
        kind, port = find_any()
        if kind == "bus":
            with Bus(port, **READ_ONLY) as bus:
                print(f"bus controller, protocol v{bus.version}: {len(bus.nodes)} node(s)")
                for n in bus.nodes:
                    line = f"  node {n['node']:2}  {n['board']:<9}  uid {n['uid']}  "
                    if not n["online"]:
                        print(line + "OFFLINE")
                        continue
                    s = bus[n["node"]]
                    skin = (f"skin {s.stored['config_id']:08X}, {s.stored['active']} taxels"
                            if s.stored else "no skin")
                    print(line + skin + ("" if s.manifest or not s.stored else
                                         "  (no manifest on this computer)"))
            return 0
        args.port = port
    # Read-only: do not assert a session on the board just to look at it.
    sensors, bus = _open(args, session=READ_ONLY)
    try:
        for s in sensors:
            info = s.info()
            print(f"board {s.board_name}, firmware API {s.caps['api_version']}, up to "
                  f"{s.caps['steps']} driven x {s.caps['channels']} sensed lines")
            if info["skin"] is None:
                print("no skin: run `opentac setup <manifest>`")
            else:
                print("skin:")
                print(_describe_skin(info["skin"]))
                print(f"  manifest    {info['manifest'] or 'NOT on this computer: run opentac setup <manifest> (or recover it from the QR code)'}")
            print("parameters (reset when the board reboots):")
            for name, p in sorted(s.params.items(), key=lambda kv: kv[1]["id"]):
                extra = f"  ({s.feedback_ohm():.0f} ohm feedback)" if name == "GAIN_SW" else ""
                print(f"  {name:<16} {p['value']:>6}   [{p['lo']}..{p['hi']}]{extra}")
    finally:
        _close(sensors, bus)
    return 0


def _start_xampling(sensors, args):
    if not args.xampling:
        return
    print("zeroing: nothing may touch the skin...")
    for s in sensors:
        z = s.zero()
        print(f"  node {s.node}: " if s.node is not None else "  ", end="")
        print(f"noise median <= {z['noise_median']}, max {z['noise_max']}, "
              f"{z['noisy']} taxels >= 100 counts")
        got = s.xampling(args.xampling, grow=args.grow)
        print(f"  xampling {'grow' if args.grow else 'fixed'}, {got} reads per frame")


def _view(args) -> int:
    from opentac.view import view

    sensors, bus = _open(args, all_nodes=True)
    try:
        _start_xampling(sensors, args)
        return view(sensors, bus=bus, shear=args.shear, vmax=args.vmax, save=args.save,
                    description=args.description, snapshot=args.snapshot)
    finally:
        _close(sensors, bus)


def _record(args) -> int:
    from opentac.record import record

    sensors, bus = _open(args)
    try:
        _start_xampling(sensors, args)
        record(sensors[0], args.dir, seconds=args.seconds, description=args.description)
    finally:
        _close(sensors, bus)
    return 0


def _flash(args) -> int:
    from opentac.device.flash import flash

    return flash(board=args.board, port=args.port, binary=args.bin, force=args.force,
                 dry_run=args.dry_run)


def _manifest_check(args) -> int:
    from opentac import manifest
    from opentac.device import boards, fits

    m = manifest.load(args.file, scan=True)
    for note in m.warnings:
        print(f"warning: {note}")
    pads = sorted(set(m.raw["rp"]) | set(m.raw["cp"]))
    print(f"{m.name}: manifest v{m.version}, {m.raw['r']} rows x {m.raw['c']} columns, "
          f"{m.n} taxels, pads {pads[0]}..{pads[-1]}, "
          f"{'shear bumps' if m.shear_groups is not None else 'no shear bumps'}, "
          f"skin id {m.config_id:08X}")
    ok = False
    titles = {name: b.title for name, b in boards().items()}
    for name, reason in fits(m).items():
        print(f"  {name:<10} {titles[name] + ': fits' if reason is None else 'no: ' + reason}")
        ok = ok or reason is None
    return 0 if ok else 1


def _manifest_from_qr(args) -> int:
    from opentac import manifest

    m = manifest.from_qr(args.text)
    if args.out:
        m.save(args.out)
        print(f"wrote {args.out}: {m.n} taxels, {m.raw['r']} x {m.raw['c']}; no outline, "
              f"edge cuts, design or shear (the QR does not carry them)")
    else:
        sys.stdout.write(m.source_bytes.decode("utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opentac", description="OpenTac: design, build, read "
                                     "and simulate custom tactile sensors.")
    parser.add_argument("--version", action="version", version=f"opentac {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    cad = sub.add_parser("cad", help="design a sensor: start the CAD tool and open it in the browser")
    cad.add_argument("--host", default="127.0.0.1", help="address to serve on (default: this computer only)")
    cad.add_argument("--port", type=int, default=None, help="port (default: first free one from 8000)")
    cad.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    cad.set_defaults(func=_cad)

    def board_args(p, nodes="a node number, `pick` to choose by LED"):
        p.add_argument("--port", help="serial port (found automatically if omitted)")
        p.add_argument("--node", help=f"a sensor behind a bus controller: {nodes}")

    def session_args(p):
        p.add_argument("--set", action="append", metavar="NAME=VALUE",
                       help="set a board parameter for this session, e.g. MUX_SETTLE_US=10 or "
                            "GAIN_SW=2 (repeatable; `opentac info` lists them)")
        p.add_argument("--imu", action="store_true", help="keep the IMU on (off by default)")
        p.add_argument("--xampling", type=int, metavar="M",
                       help="Xampling: read M taxels per frame (zeroes first: hands off)")
        p.add_argument("--grow", action="store_true",
                       help="with --xampling: M search reads, then grow around every contact")

    p = sub.add_parser("setup", help="tell the board which sensor is plugged in (once per sensor)")
    p.add_argument("manifest", help="the sensor's manifest.json or design .zip")
    board_args(p)
    p.add_argument("--dry-run", action="store_true", help="check and build the config, send nothing")
    p.add_argument("--board", help="with --dry-run: readout60, node16 or readout32")
    p.add_argument("--out", help="with --dry-run: also write the config bytes here")
    p.set_defaults(func=_setup)

    p = sub.add_parser("info", help="what is attached: board, stored sensor, parameters")
    board_args(p)
    p.set_defaults(func=_info)

    p = sub.add_parser("view", help="live heatmap (every node, on a bus controller)")
    board_args(p, nodes="a node number, `pick`, or `all` (default on a bus)")
    session_args(p)
    p.add_argument("--shear", action="store_true", help="start with shear arrows on (key s)")
    p.add_argument("--vmax", type=float, default=4095, help="top of the colour scale, counts")
    p.add_argument("--save", metavar="DIR", help="record takes into DIR: Enter starts and stops")
    p.add_argument("--description", default="", help="free text stored with each take")
    p.add_argument("--snapshot", metavar="PNG", help="save one frame to a PNG and exit")
    p.set_defaults(func=_view)

    p = sub.add_parser("record", help="record one take into DIR (Ctrl-C stops)")
    p.add_argument("dir", help="folder for takes; each take is a subfolder")
    board_args(p)
    session_args(p)
    p.add_argument("--seconds", type=float, help="stop after this long")
    p.add_argument("--description", default="", help="free text stored with the take")
    p.set_defaults(func=_record)

    p = sub.add_parser("flash", help="put the prebuilt firmware on the board (over USB, no button)")
    p.add_argument("--board", choices=["readout60", "node16", "readout32"],
                   help="which board (default: the one the running firmware reports)")
    p.add_argument("--port", help="serial port (found automatically if omitted)")
    p.add_argument("--bin", help="flash this binary instead of the prebuilt one")
    p.add_argument("--force", action="store_true", help="flash even if the board reports another name")
    p.add_argument("--dry-run", action="store_true", help="say what would happen, change nothing")
    p.set_defaults(func=_flash)

    p = sub.add_parser("manifest", help="check a manifest, or recover one from a sensor's QR code")
    msub = p.add_subparsers(dest="manifest_command", metavar="ACTION", required=True)
    q = msub.add_parser("check", help="validate a manifest and say which readout boards it fits")
    q.add_argument("file", help="manifest.json or design .zip")
    q.set_defaults(func=_manifest_check)
    q = msub.add_parser("from-qr", help="rebuild a manifest from the text of the sensor's QR code")
    q.add_argument("text", help="the text a QR scanner app reads off the back of the sensor")
    q.add_argument("-o", "--out", help="write the manifest here (default: print it)")
    q.set_defaults(func=_manifest_from_qr)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    from opentac.device.protocol import DeviceError
    from opentac.manifest import ManifestError
    try:
        return args.func(args)
    except (DeviceError, ManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
