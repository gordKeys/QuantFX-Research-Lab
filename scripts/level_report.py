"""
Which instruments actually respect support and resistance?

Runs the structure module across every exported symbol and reports the respect
rate at detected levels alongside a random-bar baseline for the same
instrument. The column that decides anything is `edge` -- respect rate minus
baseline. Everything else is context.

    python run_project.py levels --timeframe M15
    python run_project.py levels --timeframe H4 --min-touches 4

Read the output as a filter, not a ranking. Instruments with an edge near zero
do not respect levels in a way this detector can see, and no amount of entry
logic on top will fix that. Build on the ones that clear the baseline.
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import re
from pathlib import Path

import pandas as pd

from analysis.structure import build_levels, measure_reactions


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [
        (pattern.match(path.name).group(1), path)
        for path in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
        if pattern.match(path.name)
    ]


def main():
    parser = argparse.ArgumentParser(description="Measure whether instruments respect S/R levels.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--swing-lookback", type=int, default=5)
    parser.add_argument("--min-touches", type=int, default=3,
                        help="Touches required before a level counts as real.")
    parser.add_argument("--horizon", type=int, default=20,
                        help="Bars to look forward when judging a reaction.")
    parser.add_argument("--reaction-atr", type=float, default=1.0,
                        help="Move size, in ATR, that counts as a reaction.")
    parser.add_argument("--output", default="reports/level_report.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv files in {args.data_dir}/.")

    rows = []
    for symbol, path in found:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 500:
            continue

        levels = build_levels(df, lookback=args.swing_lookback)
        result = measure_reactions(
            df, levels,
            min_touches=args.min_touches,
            horizon=args.horizon,
            reaction_atr=args.reaction_atr,
        )
        if not result:
            print(f"  {symbol}: no qualifying levels")
            continue

        rows.append({
            "symbol": symbol,
            "levels": result["levels_tested"],
            "touches": result["touch_events"],
            "respect": result["respect_rate"],
            "baseline": result["baseline_rate"],
            "edge": result["edge_over_baseline"],
            "break_rate": result["break_rate"],
            "react_atr": result["median_reaction_atr"],
        })

    if not rows:
        raise SystemExit("No symbol produced qualifying levels.")

    table = pd.DataFrame(rows).sort_values("edge", ascending=False)

    display = table.copy()
    for column in ["respect", "baseline", "edge", "break_rate"]:
        display[column] = display[column].map(lambda v: f"{v:+.3f}" if column == "edge" else f"{v:.3f}")
    display["react_atr"] = display["react_atr"].map(lambda v: f"{v:.2f}")

    print(f"\n=== Level response -- {args.timeframe}, "
          f"min {args.min_touches} touches, {args.reaction_atr} ATR reaction, "
          f"{args.horizon} bar horizon ===\n")
    print(display.to_string(index=False))

    print("\nrespect  = share of level touches followed by a move away")
    print("baseline = same test at random bars on the same instrument")
    print("edge     = respect - baseline. THIS is the only column that matters.")
    print("           Near zero means the level added nothing.")

    positive = table[table["edge"] > 0.03]
    if positive.empty:
        print("\nNothing clears baseline by more than 3 points. Before concluding")
        print("S/R does not work here, try: a higher timeframe, --min-touches 4,")
        print("or a larger --reaction-atr. If the edge stays flat across all of")
        print("those, that is a real answer and worth accepting.")
    else:
        print(f"\nClearing baseline by >3 points: {', '.join(positive['symbol'])}")
        print("Build the S/R strategy on these and ignore the rest for now.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nFull table -> {output}")


if __name__ == "__main__":
    main()
