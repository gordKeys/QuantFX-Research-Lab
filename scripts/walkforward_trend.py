"""
Walkforward for daily time-series momentum, net of costs -- and benchmarked
against buy-and-hold, which is the honest bar for a long-biased trend system.

A trend follower on equity indices will look profitable simply because indices
went up. The question that matters is NOT "did it make money" -- buy-and-hold
made money too. It is:

  1. Did it beat buy-and-hold RISK-ADJUSTED? A trend system earns its keep by
     sidestepping the big drawdowns, not by out-returning a rising market. So
     the headline comparison is return/maxDD (a crude Calmar), strategy vs
     just holding the instrument over the same out-of-sample window.

  2. Did it survive walkforward and costs? Same rolling out-of-sample machinery
     that killed the intraday entry. Params chosen on prior bars only.

  3. Does the edge concentrate where the mechanism predicts? Indices and crypto
     should lead; FX majors (no structural drift) should be weak. If a random
     FX cross tops the table, the "trend effect" is really just curve-fitting.

Costs on D1 are almost irrelevant (a handful of trades a year), so unlike the
intraday tests, cost is not what decides this one -- the buy-and-hold benchmark
is.

    python run_project.py wftrend --timeframe D1
    python run_project.py wftrend --timeframe D1 --long-only
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

CLASS_COMMISSION = {"fx": 5.04, "metal": 6.00, "index": 0.0, "energy": 0.0, "crypto": 0.0}

PARAM_GRID = [
    {"entry_lookback": el, "exit_lookback": xl, "trail_atr": ta}
    for el in (30, 50, 80)
    for xl in (10, 20)
    for ta in (3.0, 5.0)
]


def classify(symbol):
    u = symbol.upper()
    if any(h in u for h in ("BTC", "ETH")):
        return "crypto"
    if any(h in u for h in ("XAU", "XAG")):
        return "metal"
    if "OIL" in u:
        return "energy"
    if any(u.startswith(h) for h in ("US30", "US500", "NAS100", "GER40", "US100")):
        return "index"
    return "fx"


def instrument_spec(data_dir, symbol, timeframe, df):
    meta_path = Path(data_dir) / f"{symbol}_{timeframe}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    point = float(meta.get("point") or 0)
    if not point:
        sample = df["close"].dropna().head(200)
        decimals = max((len(f"{v:.10f}".rstrip('0').split('.')[1]) for v in sample), default=5)
        point = 10.0 ** (-min(decimals, 5))
    tick_value = float(meta.get("trade_tick_value") or 0)
    tick_size = float(meta.get("trade_tick_size") or point)
    dollars_per_unit = tick_value / tick_size if tick_value and tick_size else None
    per_lot = None
    cmap = Path("configs/commission_map.json")
    if cmap.exists():
        entry = json.loads(cmap.read_text()).get("symbols", {}).get(symbol)
        if entry:
            per_lot = entry.get("per_lot_round_trip")
    if per_lot is None:
        per_lot = CLASS_COMMISSION.get(classify(symbol), 0.0)
    return {
        "point": point, "dollars_per_unit": dollars_per_unit,
        "commission_per_lot": per_lot,
        "volume_min": float(meta.get("volume_min") or 0.01),
        "volume_step": float(meta.get("volume_step") or 0.01),
    }


def run_trades(df, atr_values, strat, spec, risk_frac, equity_start):
    high = df["high"].to_numpy(); low = df["low"].to_numpy()
    close = df["close"].to_numpy(); open_ = df["open"].to_numpy()
    spread_price = df["spread"].to_numpy() * spec["point"]
    exit_low = pd.Series(low).rolling(strat.exit_lookback).min().shift(1).to_numpy()
    exit_high = pd.Series(high).rolling(strat.exit_lookback).max().shift(1).to_numpy()

    entries = strat.entries(df, atr_values)
    entry_at = {}
    for i, d, stop in entries:
        entry_at.setdefault(i, (d, stop))

    equity = equity_start
    trades = []
    position = None

    for i in range(len(df) - 1):
        if position is not None:
            d = position["dir"]; band = atr_values[i]
            if not np.isnan(band) and band > 0:
                if d > 0:
                    position["trail"] = max(position["trail"], high[i] - band * strat.trail_atr)
                else:
                    position["trail"] = min(position["trail"], low[i] + band * strat.trail_atr)

            # exits: initial/trailing stop, OR opposite Donchian band
            stop = position["trail"]
            hit_stop = low[i] <= stop if d > 0 else high[i] >= stop
            hit_donch = (not np.isnan(exit_low[i]) and close[i] < exit_low[i]) if d > 0 \
                        else (not np.isnan(exit_high[i]) and close[i] > exit_high[i])

            exit_price = None
            if hit_stop:
                exit_price = stop
            elif hit_donch:
                exit_price = close[i]

            if exit_price is not None:
                fill = exit_price - d * spread_price[i] / 2
                move = (fill - position["entry"]) * d
                gross = move * spec["dollars_per_unit"] * position["lots"]
                commission = spec["commission_per_lot"] * position["lots"]
                pnl = gross - commission
                equity += pnl
                trades.append({
                    "entry_bar": position["bar"], "exit_bar": i, "pnl": pnl,
                    "gross": gross, "commission": commission,
                    "R": move / position["risk_dist"] if position["risk_dist"] else 0,
                })
                position = None

        if position is None and i in entry_at and equity > 0:
            band = atr_values[i]
            if np.isnan(band) or band <= 0:
                continue
            d, init_stop = entry_at[i]
            entry = open_[i + 1] + d * spread_price[i] / 2
            risk_dist = abs(entry - init_stop)
            if risk_dist <= 0:
                continue
            lots = (equity * risk_frac) / (risk_dist * spec["dollars_per_unit"])
            step = spec["volume_step"]
            lots = max(spec["volume_min"], np.floor(lots / step) * step)
            if lots <= 0:
                continue
            position = {"bar": i + 1, "dir": d, "entry": entry, "trail": init_stop,
                        "risk_dist": risk_dist, "lots": lots}
    return trades


def metrics(trades, equity_start):
    if not trades:
        return None
    pnl = np.array([t["pnl"] for t in trades])
    eq = equity_start + np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max()
    ret = pnl.sum() / equity_start
    return {"return": ret, "maxdd": dd, "calmar": ret / dd if dd > 0 else np.inf,
            "trades": len(trades), "win_rate": (pnl > 0).mean(),
            "expectancy_R": np.mean([t["R"] for t in trades]),
            "cost": sum(t["commission"] for t in trades)}


def score(trades, equity_start):
    m = metrics(trades, equity_start)
    if not m:
        return -1e9
    return m["return"] / (1 + 5 * m["maxdd"])


def buy_hold(df, start, end, equity_start, risk_frac, spec, atr_values):
    """Hold the instrument over the OOS window, sized to a comparable initial
    risk so the return scale matches the strategy."""
    close = df["close"].to_numpy()
    i0 = start
    while i0 < end and (np.isnan(atr_values[i0]) or atr_values[i0] <= 0):
        i0 += 1
    if i0 >= end - 1:
        return None
    band = atr_values[i0]
    lots = (equity_start * risk_frac) / (band * spec["atr_stop_ref"] * spec["dollars_per_unit"]) \
        if spec["dollars_per_unit"] else 0
    lots = max(spec["volume_min"], lots)
    equity_curve = equity_start + (close[i0:end] - close[i0]) * spec["dollars_per_unit"] * lots
    peak = np.maximum.accumulate(equity_curve)
    dd = ((peak - equity_curve) / peak).max()
    ret = (equity_curve[-1] - equity_start) / equity_start
    return {"return": ret, "maxdd": dd, "calmar": ret / dd if dd > 0 else np.inf}


def walkforward(df, atr_values, spec, folds, risk_frac, equity_start, long_only):
    n = len(df)
    fold_size = n // (folds + 1)
    oos = []
    bh_returns = []
    bh_dds = []
    spec["atr_stop_ref"] = 3.0

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
        oos.extend(trades)

        bh = buy_hold(df, train_end, test_end, equity_start, risk_frac, spec, atr_values)
        if bh:
            bh_returns.append(bh["return"])
            bh_dds.append(bh["maxdd"])

    bh_agg = None
    if bh_returns:
        total_bh = np.sum(bh_returns)
        worst_dd = np.max(bh_dds)
        bh_agg = {"return": total_bh, "maxdd": worst_dd,
                  "calmar": total_bh / worst_dd if worst_dd > 0 else np.inf}
    return oos, bh_agg


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [(pattern.match(p.name).group(1), p)
            for p in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
            if pattern.match(p.name)]


def main():
    parser = argparse.ArgumentParser(description="Walkforward daily trend vs buy-and-hold.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--output", default="reports/wftrend.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    print(f"\n=== Daily-trend walkforward -- {args.timeframe}, {args.folds} folds, "
          f"{'long-only' if args.long_only else 'both sides'}, costs included ===")
    print("Donchian breakout, ATR trailing exit. Benchmarked vs buy-and-hold.")
    print("The column that matters is beat_BH: return/maxDD of strategy minus")
    print("the same for just holding it. Positive = trend timing added value.\n")
    print(f"{'symbol':10}{'class':>7}{'trades':>7}{'ret%':>8}{'maxDD%':>8}"
          f"{'calmar':>8}{'BH_ret%':>9}{'BH_dd%':>8}{'beat_BH':>9}")

    rows = []
    for symbol, path in found:
        df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 800:
            continue
        spec = instrument_spec(args.data_dir, symbol, args.timeframe, df)
        if not spec["dollars_per_unit"]:
            print(f"{symbol:10}  skipped -- no tick value in meta")
            continue
        atr_values = atr(df).to_numpy()
        oos, bh = walkforward(df, atr_values, spec, args.folds, args.risk,
                              args.equity, args.long_only)
        m = metrics(oos, args.equity)
        if not m or not bh:
            continue

        beat = m["calmar"] - bh["calmar"]
        cls = classify(symbol)
        rows.append({"symbol": symbol, "class": cls, "trades": m["trades"],
                     "return_pct": m["return"] * 100, "maxdd_pct": m["maxdd"] * 100,
                     "calmar": m["calmar"], "bh_return_pct": bh["return"] * 100,
                     "bh_dd_pct": bh["maxdd"] * 100, "beat_bh": beat,
                     "expectancy_R": m["expectancy_R"], "win_rate": m["win_rate"]})

        def fmt(x):
            return "inf" if not np.isfinite(x) else f"{x:.2f}"
        print(f"{symbol:10}{cls:>7}{m['trades']:>7}{m['return']*100:>8.1f}"
              f"{m['maxdd']*100:>8.1f}{fmt(m['calmar']):>8}{bh['return']*100:>9.1f}"
              f"{bh['maxdd']*100:>8.1f}{fmt(beat):>9}")

    if not rows:
        raise SystemExit("\nNo instrument produced OOS trades.")

    table = pd.DataFrame(rows)
    beat = table[table["beat_bh"] > 0]
    by_class = table.groupby("class")["beat_bh"].median()

    print("\n" + "=" * 74)
    print(f"Beat buy-and-hold risk-adjusted: {len(beat)}/{len(table)}")
    print(f"Median beat_BH by class:")
    for cls, val in by_class.sort_values(ascending=False).items():
        print(f"    {cls:8} {val:+.2f}  ({(table['class']==cls).sum()} instruments)")
    print("=" * 74)
    print("\nThe mechanism predicts indices and crypto lead, FX lags. If that is the")
    print("ranking, the effect is real and structural. If FX crosses top it, it is")
    print("curve-fitting. beat_BH > 0 on indices/crypto = the first defensible edge")
    print("in this project. Those are the demo candidates -- on D1, long-biased.")

    winners = beat[beat["class"].isin(["index", "crypto"])]
    if not winners.empty:
        print(f"\nDemo candidates: {', '.join(sorted(winners['symbol']))}")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
