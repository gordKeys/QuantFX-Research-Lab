"""
Tally position-management close reasons from the live jsonl logs.

Cross-platform replacement for:
    grep '"action"' logs/live_run_*.jsonl | grep -o '"action": "[a-z_]*"' | sort | uniq -c

which doesn't work on Windows cmd.exe (no grep, and '*' isn't expanded
before reaching the program the way it is in bash/zsh).

Usage:
    python scripts/action_counts.py --logs logs/live_run_*.jsonl
    python scripts/action_counts.py --logs logs/live_run_2026-07-17.jsonl
"""
from bootstrap import add_project_root
add_project_root()

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def _resolve_log_paths(patterns):
    resolved = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            resolved.extend(matches)
        elif Path(pattern).exists():
            resolved.append(pattern)
        else:
            print(f"No files matched: {pattern}")
    return sorted(set(resolved))


def main():
    parser = argparse.ArgumentParser(description="Count position_manage actions in live jsonl logs")
    parser.add_argument("--logs", nargs="+", default=["logs/live_run_*.jsonl"])
    args = parser.parse_args()

    paths = _resolve_log_paths(args.logs)
    if not paths:
        print("No log files found.")
        return

    counts = Counter()
    total_lines = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "position_manage":
                    counts[payload.get("action", "unknown")] += 1

    print(f"Scanned {len(paths)} log file(s), {total_lines} lines.")
    if not counts:
        print("No position_manage events found.")
        return

    print(f"\n{'action':>32} | count")
    print("-" * 45)
    for action, count in counts.most_common():
        print(f"{action:>32} | {count}")


if __name__ == "__main__":
    main()
