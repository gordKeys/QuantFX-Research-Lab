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
    print("\n=== QuantFX Confluence Launcher ===")
    print(f"Mode: {mode}")
    print(f"Project root: {ROOT}")
    if args.symbol:
        print(f"Symbols: {', '.join(args.symbol)}")
    if args.data:
        print(f"Data files: {', '.join(args.data)}")
    print("======================\n")


def main():
    parser = argparse.ArgumentParser(description="Confluence strategy launcher")
    parser.add_argument("mode", choices=["test", "walkforward"], help="Choose what to run")
    parser.add_argument("--symbol", action="append", help="Repeatable symbol filter")
    parser.add_argument("--data", action="append", help="Repeatable CSV path filter")
    args = parser.parse_args()

    forwarded = []
    for item in args.symbol or []:
        forwarded.extend(["--symbol", item])
    for item in args.data or []:
        forwarded.extend(["--data", item])

    print_status(args.mode, args)
    if args.mode == "test":
        return run_script("confluence_experiment.py", forwarded)
    if args.mode == "walkforward":
        return run_script("confluence_experiment.py", forwarded)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
