import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(script, extra_args=None):
    cmd = [PYTHON, str(ROOT / "scripts" / script)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd, cwd=str(ROOT))


def print_status(mode, args):
    print("\n=== QuantFX Launcher ===")
    print(f"Mode: {mode}")
    print(f"Project root: {ROOT}")
    if args.symbol:
        print(f"Symbols: {', '.join(args.symbol)}")
    if args.data:
        print(f"Data files: {', '.join(args.data)}")
    if mode == "live":
        print(f"Dry run: {args.dry_run}")
        print(f"Loop once: {args.loop_once}")
        print(f"Max consecutive losses: {args.max_consecutive_losses or 3}")
    print("======================\n")


def main():
    parser = argparse.ArgumentParser(description="QuantFX project launcher")
    parser.add_argument(
        "mode",
        choices=["test", "walkforward", "sweep", "combo", "live", "milestone", "tournament"],
        help="Choose what to run",
    )
    parser.add_argument("--symbol", action="append", help="Repeatable symbol filter")
    parser.add_argument("--data", action="append", help="Repeatable CSV path filter")
    parser.add_argument("--dry-run", action="store_true", help="Live mode only")
    parser.add_argument("--loop-once", action="store_true", help="Live mode only")
    parser.add_argument("--max-consecutive-losses", type=int, help="Live mode only")
    args = parser.parse_args()

    forwarded = []
    for item in args.symbol or []:
        forwarded.extend(["--symbol", item])
    for item in args.data or []:
        forwarded.extend(["--data", item])

    if args.mode == "test":
        print_status(args.mode, args)
        return run_script("evaluate_multi_symbol.py", forwarded)

    if args.mode == "walkforward":
        print_status(args.mode, args)
        return run_script("walkforward_multi_symbol.py", forwarded)

    if args.mode == "sweep":
        print_status(args.mode, args)
        return run_script("sweep_mean_reversion.py", forwarded)

    if args.mode == "combo":
        print_status(args.mode, args)
        return run_script("run_symbol_combo.py", forwarded)

    if args.mode == "live":
        print_status(args.mode, args)
        live_args = forwarded[:]
        if args.dry_run:
            live_args.append("--dry-run")
        if args.loop_once:
            live_args.append("--loop-once")
        if args.max_consecutive_losses is not None:
            live_args.extend(["--max-consecutive-losses", str(args.max_consecutive_losses)])
        if not live_args:
            live_args = ["--symbols", "EURUSD", "GBPUSD", "XAUUSD"]
        return run_script("live_runner.py", live_args)

    if args.mode == "milestone":
        print_status(args.mode, args)
        return run_script("milestone_report.py", forwarded)

    if args.mode == "tournament":
        print_status(args.mode, args)
        return run_script("tournament_report.py", forwarded)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
