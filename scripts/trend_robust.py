"""
Trend robustness -- is the daily-trend edge real, or a few lucky big trends?

The walkforward said crypto and metals crush buy-and-hold and the US indices
beat it modestly. But the crypto/metal numbers rest on suspiciously few trades
(BTCUSD: ~50 over 15 years) and one worry dominates: a Calmar of 30 built on
4-5 parabolic runs is not an edge, it is a small sample that happened to contain
Bitcoin's history. This script stress-tests exactly that.

Two cuts, both aimed at "concentrated in a few trends / one regime":

  CONCENTRATION   How much of the total profit comes from the single best
                  trade, and from the top 3? If one trade is 60% of the P&L,
                  the "edge" is one lucky trend wearing a track record. Reported
                  as top1_share and top3_share, plus the count of winning trades
                  that actually carry the result.

  TIME SPLIT      Split the out-of-sample trades into first-half and second-half
                  by calendar time. A real structural edge shows up in BOTH
                  halves. An edge that only appears in one half is a regime that
                  happened once -- most dangerous when the good half is the
                  recent one, because that is what a naive backtest rewards.

The verdict combines them: an instrument is "robust" only if no single trade
dominates (top1 < ~40%), enough winners contribute (>= 8), AND both time halves
are positive. Anything failing these is a watchlist name at best -- tradeable
only tiny, with the understanding that its history is a handful of trends.

    python run_project.py trend_robust --timeframe D1 --long-only
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.daily_trend import DailyTrend, atr
from walkforward_trend import (
    CLASS_COMMISSION, PARAM_GRID, classify, instrument_spec, run_trades, score,
)


def walkforward_trades(df, atr_values, spec, folds, risk_frac, equity_start, long_only):
    """Same rolling walkforward as wftrend, but returns trades tagged with the
    calendar time of entry so we can split by period."""
    n = len(df)
    fold_size = n // (folds + 1)
    oos = []
    index = df.index

    for k in range(1, folds + 1):
        train_end = fold_size * k
        test_end = fold_size * (k + 1) if k < folds else n

        best_params, best_s = None, -1e18
        for params in PARAM_GRID:
            strat = DailyTrend(**params, long_only=long_only)
            trades = run_trades(df.iloc[:train_end], atr_values[:train_end], strat,
                                spec, risk_frac, equity_start)
            s = score(trades, equity_start)
            if s > best_s:
                best_s, best_params = s, params
        if best_params is None:
            continue

        strat = DailyTrend(**best_params, long_only=long_only)
        test_df = df.iloc[train_end:test_end]
        test_atr = atr_values[train_end:test_end]
        trades = run_trades(test_df, test_atr, strat, spec, risk_frac, equity_start)
        # tag each trade with its entry timestamp for the time split
        base = train_end
        for t in trades:
            t["entry_time"] = index[min(base + t["entry_bar"], n - 1)]
        oos.extend(trades)
    return oos


def analyse(trades):
    if len(trades) < 5:
        return None
    pnl = np.array([t["pnl"] for t in trades])
    total = pnl.sum()
    wins = pnl[pnl > 0]

    # concentration
    ordered = np.sort(pnl)[::-1]
    # Basis is total profit when positive; otherwise gross winning P&L, so the
    # share stays meaningful even for net-losing instruments.
    basis = total if total > 0 else (wins.sum() if wins.size else np.nan)
    top1 = ordered[0] / basis if basis and basis > 0 else np.nan
    top3 = ordered[:3].sum() / basis if basis and basis > 0 else np.nan

    # time split by entry order (already chronological within folds, but sort to be safe)
    times = np.array([t["entry_time"] for t in trades])
    order = np.argsort(times)
    pnl_sorted = pnl[order]
    mid = len(pnl_sorted) // 2
    first_half = pnl_sorted[:mid].sum()
    second_half = pnl_sorted[mid:].sum()

    return {
        "trades": len(trades),
        "winners": int((pnl > 0).sum()),
        "total_pnl": total,
        "top1_share": top1,
        "top3_share": top3,
        "first_half_pnl": first_half,
        "second_half_pnl": second_half,
        "both_halves_positive": (first_half > 0) and (second_half > 0),
    }


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [(pattern.match(p.name).group(1), p)
            for p in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
            if pattern.match(p.name)]


def main():
    parser = argparse.ArgumentParser(description="Stress-test the daily-trend edge for concentration and regime.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--output", default="reports/trend_robust.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    print(f"\n=== Trend robustness -- {args.timeframe}, "
          f"{'long-only' if args.long_only else 'both sides'} ===")
    print("Is the edge broad, or a few big trends in one regime?\n")
    print(f"{'symbol':10}{'class':>7}{'winners':>8}{'top1%':>7}{'top3%':>7}"
          f"{'1stHalf':>9}{'2ndHalf':>9}{'robust':>8}")

    rows = []
    for symbol, path in found:
        df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 800:
            continue
        spec = instrument_spec(args.data_dir, symbol, args.timeframe, df)
        if not spec["dollars_per_unit"]:
            continue
        atr_values = atr(df).to_numpy()
        trades = walkforward_trades(df, atr_values, spec, args.folds, args.risk,
                                    args.equity, args.long_only)
        a = analyse(trades)
        if not a:
            continue

        cls = classify(symbol)
        robust = (
            a["both_halves_positive"]
            and a["total_pnl"] > 0
            and not np.isnan(a["top1_share"]) and a["top1_share"] < 0.40
            and a["winners"] >= 8
        )
        rows.append({"symbol": symbol, "class": cls, **a, "robust": robust})

        t1 = "n/a" if np.isnan(a["top1_share"]) else f"{a['top1_share']*100:.0f}%"
        t3 = "n/a" if np.isnan(a["top3_share"]) else f"{a['top3_share']*100:.0f}%"
        print(f"{symbol:10}{cls:>7}{a['winners']:>8}{t1:>7}{t3:>7}"
              f"{a['first_half_pnl']:>+9.0f}{a['second_half_pnl']:>+9.0f}"
              f"{'YES' if robust else 'no':>8}")

    if not rows:
        raise SystemExit("No instrument produced enough trades.")

    table = pd.DataFrame(rows)
    robust = table[table["robust"]]

    print("\n" + "=" * 72)
    print("A result is 'robust' only if: both time-halves positive, top single")
    print("trade < 40% of profit, and at least 8 winning trades. This is what")
    print("separates a real edge from a few lucky trends.\n")
    if robust.empty:
        print("NOTHING passes. Every positive result so far leans on a handful of")
        print("trends or one regime. That does not mean the indices finding is")
        print("false -- it means the daily sample is too short to confirm it, which")
        print("is itself the honest answer: trend following needs decades to prove")
        print("out, and you have a few. Treat any demo as watchlist-sized.")
    else:
        keep = ', '.join(f"{r.symbol}({r['class']})" for _, r in robust.iterrows())
        print(f"ROBUST (broad + both regimes): {keep}")
        print("\nThese are the only names whose edge does not rest on a few trends.")
        print("Demo-worthy, long-biased, D1, tiny size. Still trend-following, so")
        print("expect long flat stretches between trades.")

    print("=" * 72)
    print("\nReminder on the big crypto/metal Calmars from wftrend: high top1_share")
    print("or a failed half-split there means the number was one trend, not skill.")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
