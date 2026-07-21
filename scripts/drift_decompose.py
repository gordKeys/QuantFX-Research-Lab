"""
Drift decomposition -- how much of the dip-entry result is just direction?

The sequential backtest showed 21/21 instruments profitable out-of-sample. That
is not what a real intraday edge looks like; it is what riding a directional
trend looks like when every instrument in the book trended.

The dip-entry signal enters SHORT when price rises into a zone and LONG when it
falls into one. Over 2023-2026 most of these instruments had strong directional
drift (dollar moves, gold up, crypto up). A signal that leans one direction wins
in a market that drifts that way -- from the drift, not from timing.

The original barrier control could not see this because it assigned control
directions by coin flip. A coin-flip control in a trending market sits at 0.50
while a direction-leaning signal sits above it, and the gap looks like edge when
it is only the signal's directional tilt.

This script separates the two with three matched measurements per instrument:

  signal            the dip-entry rule as traded
  same_dir_random   SAME direction labels, RANDOM entry times -> holds drift
                    constant, varies only timing
  buy_hold_dir      same-direction buy-and-hold over each trade's window -> pure
                    drift, no timing at all

Reading:
  signal >> same_dir_random  -> entry TIMING has value beyond direction
  signal ~= same_dir_random  -> only direction mattered, i.e. it is a trend bet
  buy_hold_dir already high  -> the direction itself came from drift

    python run_project.py drift --timeframe H4 --barrier 3.0
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.hypotheses import _find_fvgs, _zone_events
from analysis.structure import atr


def resolve(df, entries, k, horizon):
    a = atr(df).to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
    o = df["open"].to_numpy(); c = df["close"].to_numpy()
    wins = n = 0; bh_wins = 0
    for i, d in entries:
        if i + 1 >= len(h):
            continue
        band = a[i]
        if np.isnan(band) or band <= 0:
            continue
        entry = o[i + 1]; tp = entry + d * band * k; sl = entry - d * band * k
        end = min(i + 1 + horizon, len(h)); res = None; jlast = i + 1
        for j in range(i + 1, end):
            jlast = j
            htp = h[j] >= tp if d > 0 else l[j] <= tp
            hsl = l[j] <= sl if d > 0 else h[j] >= sl
            if htp and hsl: res = False; break
            if htp: res = True; break
            if hsl: res = False; break
        if res is None:
            continue
        n += 1; wins += int(res)
        if (c[jlast] - entry) * d > 0:
            bh_wins += 1
    return (wins / n if n else 0), (bh_wins / n if n else 0), n


def main():
    parser = argparse.ArgumentParser(description="Decompose dip-entry into timing vs drift.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--barrier", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--output", default="reports/drift_decompose.csv")
    args = parser.parse_args()

    pattern = re.compile(rf"^(.+)_{re.escape(args.timeframe)}\.csv$")
    found = [(pattern.match(p.name).group(1), p)
             for p in sorted(Path(args.data_dir).glob(f"*_{args.timeframe}.csv"))
             if pattern.match(p.name)]
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    rng = np.random.default_rng(0)
    print(f"\n=== Drift decomposition -- {args.timeframe}, {args.barrier} ATR ===\n")
    print(f"{'symbol':10}{'drift%':>8}{'signal':>8}{'sameDir_rand':>14}"
          f"{'timing_gap':>12}{'BH_dir%':>9}")

    rows = []
    for symbol, path in found:
        df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 2000:
            continue
        a = atr(df).to_numpy()
        ev = _zone_events(df, a, _find_fvgs(df, a))
        if len(ev) < 50:
            continue

        c = df["close"].to_numpy(); drift = (c[-1] - c[0]) / c[0] * 100
        sig, bh, n = resolve(df, ev, args.barrier, args.horizon)

        valid = np.arange(30, len(df) - args.horizon - 1)
        dirs = [d for _, d in ev]
        rand = list(zip(rng.choice(valid, size=len(dirs), replace=False), dirs))
        rnd, _, _ = resolve(df, rand, args.barrier, args.horizon)

        rows.append({"symbol": symbol, "drift": drift, "signal": sig,
                     "same_dir_random": rnd, "timing_gap": sig - rnd, "bh_dir": bh})
        print(f"{symbol:10}{drift:>+8.1f}{sig:>8.3f}{rnd:>14.3f}"
              f"{sig-rnd:>+12.3f}{bh*100:>9.0f}")

    table = pd.DataFrame(rows)
    print("\n" + "=" * 64)
    print(f"median signal win rate       : {table['signal'].median():.3f}")
    print(f"median same-direction random : {table['same_dir_random'].median():.3f}")
    print(f"median TIMING gap            : {table['timing_gap'].median():+.3f}")
    print(f"median buy&hold-in-direction : {table['bh_dir'].median():.3f}")
    print("=" * 64)
    gap = table["timing_gap"].median()
    if gap > 0.03:
        print("\nTiming gap is positive and material: entering at the dip beats")
        print("entering at a random time in the SAME direction. There is residual")
        print("value in the timing, separate from the drift. Worth pursuing -- but")
        print("size it against buy&hold, which is the honest benchmark now.")
    else:
        print("\nTiming gap is ~zero: once direction is held constant, WHEN you")
        print("enter does not matter. The result was a directional/trend bet that")
        print("happened to pay because the sample trended. Not a timing edge.")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
