"""
Approach 1 (carry) + Approach 3 (neglected corners) -- structural, not predictive.

CARRY. Positive-swap positions earn rollover every night just for being held.
The question is whether the swap income is meaningful RELATIVE to the price risk
you take to earn it -- carry only works when the yield is large versus how much
the pair moves against you. This needs swap_long / swap_short from the metadata,
which the exporter now captures. Without it, carry cannot be assessed and the
probe says so rather than guessing.

SESSION STRUCTURE. Approach 3 says hunt where funds do not bother. One concrete
version: most automated flow concentrates in London and NY. The Asian session,
session opens, and specific hours are less mined. This probe slices each
instrument by hour and looks for hours where SOMETHING is structurally
different -- a directional bias, or a volatility concentration -- that a
session-scoped rule could exploit. It is descriptive: it finds candidates, it
does not confirm edges. Anything it flags still has to survive the same
walkforward everything else did.

The honesty rule stays: an hour that looks directional gets checked against how
many independent days contributed, because 24 hourly buckets on a trending
sample will always show SOME hour looking biased by chance.

    python run_project.py carry_probe --timeframe H4
    python run_project.py carry_probe --timeframe M15 --session-hours
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def load_meta(data_dir, symbol, timeframe):
    path = Path(data_dir) / f"{symbol}_{timeframe}_meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def carry_assessment(df, meta):
    """
    Compare nightly swap income to daily price volatility, both in points.

    swap is quoted in points per lot per night by most brokers. Daily ATR in
    points is the scale of daily price risk. carry_ratio = swap / daily_range
    says how many nights of carry one average adverse day wipes out. A ratio
    near or above 1 is a real carry instrument; 0.01 means the swap is a
    rounding error against the price risk and carry is not a strategy here.
    """
    point = meta.get("point")
    swap_long = meta.get("swap_long")
    swap_short = meta.get("swap_short")
    if point is None or swap_long is None or swap_short is None:
        return None

    daily_range_points = float(atr(df).median()) / point if point else np.nan
    best_swap = max(swap_long, swap_short)  # the tradeable side
    direction = "long" if swap_long >= swap_short else "short"
    return {
        "swap_long": swap_long,
        "swap_short": swap_short,
        "best_side": direction,
        "best_swap_points": best_swap,
        "daily_range_points": daily_range_points,
        "carry_ratio": best_swap / daily_range_points if daily_range_points else np.nan,
    }


def session_structure(df):
    """
    Per-hour directional bias and volatility share.

    For each hour of the (broker server) day: mean forward 1-bar return, its
    t-stat across days, and the hour's share of total daily range. The t-stat
    guards against a single big day masquerading as an hourly edge.
    """
    df = df.copy()
    df["ret"] = df["close"].pct_change().shift(-1)  # forward 1-bar return
    df["range"] = df["high"] - df["low"]
    df["hour"] = df.index.hour

    rows = []
    total_range = df["range"].sum()
    for hour, block in df.groupby("hour"):
        r = block["ret"].dropna()
        if len(r) < 50:
            continue
        t = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() else 0
        rows.append({
            "hour": hour,
            "mean_ret_bps": r.mean() * 1e4,
            "t_stat": t,
            "range_share": block["range"].sum() / total_range if total_range else 0,
            "n": len(r),
        })
    return pd.DataFrame(rows)


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [(pattern.match(p.name).group(1), p)
            for p in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
            if pattern.match(p.name)]


def main():
    parser = argparse.ArgumentParser(description="Probe carry and session-structure edges.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--session-hours", action="store_true",
                        help="Also print the per-hour session structure table.")
    parser.add_argument("--output", default="reports/carry_probe.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    print(f"\n=== Carry + session probe -- {args.timeframe} ===\n")
    print("CARRY (needs swap data in meta):")
    print(f"{'symbol':10}{'swap_long':>11}{'swap_short':>11}{'side':>6}"
          f"{'carry_ratio':>13}")

    carry_rows = []
    missing_swap = []
    for symbol, path in found:
        df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 500:
            continue
        meta = load_meta(args.data_dir, symbol, args.timeframe)
        ca = carry_assessment(df, meta)
        if ca is None:
            missing_swap.append(symbol)
            continue
        carry_rows.append({"symbol": symbol, **ca})
        print(f"{symbol:10}{ca['swap_long']:>11.2f}{ca['swap_short']:>11.2f}"
              f"{ca['best_side']:>6}{ca['carry_ratio']:>13.4f}")

    if missing_swap:
        print(f"\n  No swap data for: {', '.join(missing_swap)}")
        print("  Re-export with export_structure_data.py to capture swap_long/short.")

    if carry_rows:
        ct = pd.DataFrame(carry_rows).sort_values("carry_ratio", ascending=False)
        best = ct.iloc[0]
        print(f"\n  Best carry ratio: {best['symbol']} at {best['carry_ratio']:.4f} "
              f"({best['best_side']} side)")
        print("  carry_ratio > ~0.05 is worth a look; below that the swap is noise")
        print("  against the price risk. Carry pairs are held for WEEKS, not traded.")

    if args.session_hours:
        print("\n\nSESSION STRUCTURE (per hour, server time):")
        for symbol, path in found[:6]:  # cap output; run per-symbol for detail
            df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
            if len(df) < 1000:
                continue
            ss = session_structure(df)
            if ss.empty:
                continue
            strong = ss[ss["t_stat"].abs() > 3].sort_values("t_stat", key=abs, ascending=False)
            peak_vol = ss.sort_values("range_share", ascending=False).iloc[0]
            print(f"\n  {symbol}:")
            print(f"    busiest hour: {int(peak_vol['hour']):02d}:00 "
                  f"({peak_vol['range_share']*100:.0f}% of daily range)")
            if not strong.empty:
                for _, r in strong.head(3).iterrows():
                    print(f"    hour {int(r['hour']):02d}:00  bias {r['mean_ret_bps']:+.1f}bps  "
                          f"t={r['t_stat']:+.1f}  (n={int(r['n'])})")
            else:
                print("    no hour with |t| > 3 -- no directional session edge here")
        print("\n  |t| > 3 flags an hour worth testing with a session-scoped rule.")
        print("  It is a CANDIDATE, not an edge -- it still has to survive a")
        print("  walkforward. Treat a lone significant hour with suspicion: 24")
        print("  buckets guarantee a few look interesting by chance.")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if carry_rows:
        pd.DataFrame(carry_rows).to_csv(out, index=False)
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
