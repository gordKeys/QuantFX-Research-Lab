"""
Run the pre-registered structure hypothesis battery.

    python run_project.py battery --timeframe M15
    python run_project.py battery --timeframe H4

Thresholds are fixed in analysis/hypotheses.py and are not tunable from the
command line on purpose. Being able to slide the significance bar after seeing
the output is how the S/R report produced two winners that were noise.
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import re
from pathlib import Path

import pandas as pd

from analysis.hypotheses import HYPOTHESES, THRESHOLD_SIGMA, pool, test_hypothesis


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [
        (pattern.match(path.name).group(1), path)
        for path in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
        if pattern.match(path.name)
    ]


def main():
    parser = argparse.ArgumentParser(description="Pre-registered structure hypothesis battery.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--reaction-atr", type=float, default=1.0)
    parser.add_argument("--output", default="reports/hypothesis_battery.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv files in {args.data_dir}/.")

    frames = {}
    for symbol, path in found:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        frames[symbol] = df.set_index("time").sort_index()

    all_rows = []
    print(f"\n=== Hypothesis battery -- {args.timeframe}, "
          f"{args.reaction_atr} ATR barriers, {args.horizon} bar horizon ===")
    print(f"Pre-registered threshold: {THRESHOLD_SIGMA} sigma, "
          f"consistent sign across timeframes required.\n")

    for name, (_, claim) in HYPOTHESES.items():
        results = []
        for symbol, df in frames.items():
            if len(df) < 1000:
                continue
            result = test_hypothesis(df, name, args.horizon, args.reaction_atr)
            if result:
                result["symbol"] = symbol
                result["hypothesis"] = name
                results.append(result)

        if not results:
            print(f"{name}: not enough events on any instrument\n")
            continue

        pooled = pool(results)
        hits = [r for r in results if r["sigma"] >= THRESHOLD_SIGMA]
        misses = [r for r in results if r["sigma"] <= -THRESHOLD_SIGMA]

        verdict = "PASS" if pooled and abs(pooled["sigma"]) >= THRESHOLD_SIGMA else "flat"
        print(f"--- {name}: \"{claim}\" ---")
        print(f"  pooled  {pooled['rate']:.3f} vs control {pooled['control']:.3f}  "
              f"edge {pooled['edge']:+.4f}  ({pooled['sigma']:+.1f} sigma, "
              f"{pooled['events']} events)  -> {verdict}")
        print(f"  per-instrument at {THRESHOLD_SIGMA} sigma: "
              f"{len(hits)} positive, {len(misses)} negative, "
              f"{len(results)} tested")
        if hits:
            print(f"    positive: {', '.join(sorted(r['symbol'] for r in hits))}")
        if misses:
            print(f"    negative: {', '.join(sorted(r['symbol'] for r in misses))}")
        if hits and misses:
            print("    NOTE: significant results in both directions -- "
                  "that pattern is noise, not an effect.")
        print()

        all_rows.extend(results)

    if all_rows:
        table = pd.DataFrame(all_rows)[
            ["hypothesis", "symbol", "events", "rate", "control", "edge", "err", "sigma"]
        ].sort_values(["hypothesis", "sigma"], ascending=[True, False])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        print(f"Per-instrument detail -> {output}")
        print("\nRun the other timeframe before drawing any conclusion. A hypothesis")
        print("counts as live only if the pooled result clears threshold with the")
        print("same sign on BOTH M15 and H4.")


if __name__ == "__main__":
    main()
