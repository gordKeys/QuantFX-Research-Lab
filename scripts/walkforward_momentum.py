"""
Walkforward for the momentum-continuation strategy, net of real costs.

This is the test that decides whether anything in this project reaches a demo
account. Everything it does is chosen to make a false positive hard:

  ROLLING WALKFORWARD   History is cut into N folds. Each fold's parameters are
                        chosen on the bars BEFORE it and scored only on the fold
                        itself. A strategy that only works when it can see the
                        whole sample dies here. We report the out-of-sample
                        folds stitched together -- never the in-sample fit.

  REAL EXIT SIMULATION  Bar-by-bar trailing stop. Entry at next bar open plus
                        half the spread. The structural stop and the trailing
                        stop both live here. If both the stop and a new trail
                        level are touched in one bar, the stop wins (pessimistic
                        intrabar assumption).

  FULL COST MODEL       Every trade pays: half-spread on entry, half-spread on
                        exit, and measured per-lot commission from
                        commission_map.json (falling back to asset-class
                        defaults). Costs are charged in account currency using
                        the instrument's real tick value, so a pip on EURUSD and
                        a point on US30 are compared in dollars, not raw price.

  POSITION SIZING       Fixed fraction of equity risked across the structural
                        stop distance. Skips any instrument whose metadata lacks
                        tick value rather than guessing contract size.

  SYMBOL SELECTION      Reports each instrument separately so you can keep the
                        ones that clear a bar and drop the rest, rather than
                        trading the whole book because the median looked fine.

Output columns per instrument:
  oos_return   out-of-sample return %, costs deducted
  oos_dd       out-of-sample max drawdown %
  trades/day   realised frequency
  win%         hit rate under the trailing exit
  profit_factor gross win $ / gross loss $
  cost_drag%   costs as a share of gross profit -- how hard fees bite this pair
  expectancy_R average R per trade after costs

    python run_project.py walkforward --timeframe H4
    python run_project.py walkforward --timeframe M15 --folds 6 --risk 0.005
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.momentum_continuation import MomentumContinuation, atr

CLASS_COMMISSION = {"fx": 5.04, "metal": 6.00, "index": 0.0, "energy": 0.0, "crypto": 0.0}

# Parameter grid searched IN-SAMPLE on each fold. Kept small on purpose -- a big
# grid finds spurious optima. These are the two knobs that actually change
# behaviour: how big an impulse to require, and how loose to trail.
PARAM_GRID = [
    {"impulse_atr": ia, "trail_atr": ta}
    for ia in (0.8, 1.0, 1.5)
    for ta in (1.5, 2.5, 4.0)
]


def classify(symbol):
    u = symbol.upper()
    if any(h in u for h in ("BTC", "ETH")):
        return "crypto"
    if any(h in u for h in ("XAU", "XAG")):
        return "metal"
    if "OIL" in u:
        return "energy"
    if any(u.startswith(h) for h in ("US30", "US500", "NAS100", "GER40")):
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
        payload = json.loads(cmap.read_text())
        entry = payload.get("symbols", {}).get(symbol)
        if entry:
            per_lot = entry.get("per_lot_round_trip")
    if per_lot is None:
        per_lot = CLASS_COMMISSION.get(classify(symbol), 0.0)

    return {
        "point": point,
        "dollars_per_unit": dollars_per_unit,
        "commission_per_lot": per_lot,
        "volume_min": float(meta.get("volume_min") or 0.01),
        "volume_step": float(meta.get("volume_step") or 0.01),
    }


def run_trades(df, atr_values, entries, params, spec, risk_frac, equity_start):
    """
    Simulate the trailing-stop exit bar by bar. Returns a list of trade dicts.
    One position at a time; overlapping entries are skipped.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    spread_price = df["spread"].to_numpy() * spec["point"]

    trail_atr = params["trail_atr"]
    pad = params.get("stop_pad_atr", 0.1)

    entry_at = {}
    for i, direction, struct_stop in entries:
        entry_at.setdefault(i, (direction, struct_stop))

    equity = equity_start
    trades = []
    position = None

    for i in range(len(df) - 1):
        if position is not None:
            direction = position["dir"]
            band = atr_values[i]

            # advance the trailing stop in the trade's favour
            if not np.isnan(band) and band > 0:
                if direction > 0:
                    position["trail"] = max(position["trail"], high[i] - band * trail_atr)
                else:
                    position["trail"] = min(position["trail"], low[i] + band * trail_atr)

            stop = position["stop"] if not position["armed"] else position["trail"]

            hit = low[i] <= stop if direction > 0 else high[i] >= stop
            if hit:
                exit_fill = stop - direction * spread_price[i] / 2
                move = (exit_fill - position["entry"]) * direction
                gross = move * spec["dollars_per_unit"] * position["lots"]
                commission = spec["commission_per_lot"] * position["lots"]
                pnl = gross - commission
                equity += pnl
                r_multiple = move / position["risk_dist"] if position["risk_dist"] else 0
                trades.append({
                    "entry_bar": position["bar"], "exit_bar": i, "dir": direction,
                    "lots": position["lots"], "pnl": pnl, "gross": gross,
                    "commission": commission, "equity": equity, "R": r_multiple,
                })
                position = None
            else:
                # arm the trailing stop once price has moved 1R in favour
                move_now = (high[i] - position["entry"]) if direction > 0 else (position["entry"] - low[i])
                if not position["armed"] and move_now >= position["risk_dist"]:
                    position["armed"] = True

        if position is None and i in entry_at and equity > 0:
            band = atr_values[i]
            if np.isnan(band) or band <= 0:
                continue
            direction, struct_stop = entry_at[i]
            entry = open_[i + 1] + direction * spread_price[i] / 2
            stop = struct_stop - direction * band * pad
            risk_dist = abs(entry - stop)
            if risk_dist <= 0:
                continue
            lots = (equity * risk_frac) / (risk_dist * spec["dollars_per_unit"])
            step = spec["volume_step"]
            lots = max(spec["volume_min"], np.floor(lots / step) * step)
            if lots <= 0:
                continue
            position = {
                "bar": i + 1, "dir": direction, "entry": entry, "stop": stop,
                "trail": stop, "risk_dist": risk_dist, "lots": lots, "armed": False,
            }

    return trades


def score(trades, equity_start):
    """In-sample scoring metric for parameter selection: return / (1 + maxDD)."""
    if not trades:
        return -1e9
    pnl = np.array([t["pnl"] for t in trades])
    equity = equity_start + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = ((peak - equity) / peak).max()
    ret = pnl.sum() / equity_start
    return ret / (1 + 10 * dd)     # penalise drawdown heavily


def summarise(trades, equity_start, days):
    if not trades:
        return None
    pnl = np.array([t["pnl"] for t in trades])
    gross = np.array([t["gross"] for t in trades])
    equity = equity_start + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    gross_win = gross[gross > 0].sum()
    total_cost = sum(t["commission"] for t in trades) + (gross - pnl).sum()
    return {
        "trades": len(trades),
        "per_day": len(trades) / max(days, 1),
        "win_rate": (pnl > 0).mean(),
        "return_pct": pnl.sum() / equity_start * 100,
        "max_dd_pct": dd.max() * 100,
        "profit_factor": wins / losses if losses else np.inf,
        "expectancy_R": np.mean([t["R"] for t in trades]),
        "cost_drag_pct": (total_cost / gross_win * 100) if gross_win > 0 else np.nan,
        "total_cost": total_cost,
    }


def walkforward(df, atr_values, strat_params_grid, spec, folds, risk_frac, equity_start):
    """
    Rolling walkforward. Fold k trains on everything before it, tests on itself.
    Returns the stitched out-of-sample trades and the chosen params per fold.
    """
    n = len(df)
    fold_size = n // (folds + 1)   # first block is train-only
    oos_trades = []
    chosen = []

    base = MomentumContinuation()

    for k in range(1, folds + 1):
        train_end = fold_size * k
        test_end = fold_size * (k + 1) if k < folds else n

        train_df = df.iloc[:train_end]
        train_atr = atr_values[:train_end]

        best_params, best_score = None, -1e18
        for params in strat_params_grid:
            strat = MomentumContinuation(**{**base.as_dict(), **params})
            entries = [e for e in strat.entries(train_df, train_atr)]
            trades = run_trades(train_df, train_atr, entries, {**base.as_dict(), **params},
                                spec, risk_frac, equity_start)
            s = score(trades, equity_start)
            if s > best_score:
                best_score, best_params = s, params

        if best_params is None:
            continue

        # test on the held-out fold with the chosen params
        test_slice = slice(train_end, test_end)
        test_df = df.iloc[test_slice]
        test_atr = atr_values[test_slice]
        strat = MomentumContinuation(**{**base.as_dict(), **best_params})
        # entries must be found within the test window only
        entries = strat.entries(test_df, test_atr)
        trades = run_trades(test_df, test_atr, entries, {**base.as_dict(), **best_params},
                            spec, risk_frac, equity_start)
        oos_trades.extend(trades)
        chosen.append(best_params)

    return oos_trades, chosen


def main():
    parser = argparse.ArgumentParser(description="Cost-aware walkforward of momentum continuation.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--output", default="reports/walkforward.csv")
    args = parser.parse_args()

    pattern = re.compile(rf"^(.+)_{re.escape(args.timeframe)}\.csv$")
    found = [(pattern.match(p.name).group(1), p)
             for p in sorted(Path(args.data_dir).glob(f"*_{args.timeframe}.csv"))
             if pattern.match(p.name)]
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    print(f"\n=== Walkforward -- {args.timeframe}, {args.folds} folds, "
          f"{args.risk*100:.1f}% risk/trade, costs included ===")
    print("Momentum-continuation entry, structural stop, ATR trailing exit.")
    print("Every row is OUT-OF-SAMPLE: params chosen on prior bars only.\n")
    print(f"{'symbol':10}{'trades':>7}{'/day':>6}{'win%':>6}{'oos_ret%':>9}"
          f"{'maxDD%':>8}{'PF':>6}{'expR':>7}{'cost_drag%':>11}")

    rows = []
    for symbol, path in found:
        df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 3000:
            continue

        spec = instrument_spec(args.data_dir, symbol, args.timeframe, df)
        if not spec["dollars_per_unit"]:
            print(f"{symbol:10}  skipped -- no tick value in meta")
            continue

        atr_values = atr(df).to_numpy()
        oos_trades, chosen = walkforward(df, atr_values, PARAM_GRID, spec,
                                         args.folds, args.risk, args.equity)
        if not oos_trades:
            continue

        span_days = (df.index.max() - df.index.min()).days * (args.folds / (args.folds + 1))
        stats = summarise(oos_trades, args.equity, span_days)
        if not stats:
            continue
        stats["symbol"] = symbol
        rows.append(stats)

        pf = stats["profit_factor"]
        pf_str = "inf" if not np.isfinite(pf) else f"{pf:.2f}"
        drag = stats["cost_drag_pct"]
        drag_str = "n/a" if np.isnan(drag) else f"{drag:.0f}%"
        print(f"{symbol:10}{stats['trades']:>7}{stats['per_day']:>6.2f}"
              f"{stats['win_rate']*100:>6.0f}{stats['return_pct']:>9.1f}"
              f"{stats['max_dd_pct']:>8.1f}{pf_str:>6}{stats['expectancy_R']:>7.2f}{drag_str:>11}")

    if not rows:
        raise SystemExit("\nNo instrument produced out-of-sample trades.")

    table = pd.DataFrame(rows)
    winners = table[(table["return_pct"] > 0) & (table["profit_factor"] > 1.1)
                    & (table["expectancy_R"] > 0)]

    print("\n" + "=" * 72)
    print(f"{len(table)} instruments tested out-of-sample, costs deducted.")
    print(f"Profitable after costs: {(table['return_pct'] > 0).sum()}/{len(table)}")
    print(f"Median OOS return: {table['return_pct'].median():+.1f}%   "
          f"Median maxDD: {table['max_dd_pct'].median():.1f}%   "
          f"Median expectancy: {table['expectancy_R'].median():+.2f}R")
    print(f"Median cost drag: {table['cost_drag_pct'].median():.0f}% of gross profit")
    if not winners.empty:
        keep = ', '.join(sorted(winners['symbol']))
        print(f"\nCLEAR THE BAR (return>0, PF>1.1, expectancy>0):\n  {keep}")
        print(f"\nCombined trades/day on that set: "
              f"{winners['per_day'].sum():.1f}")
        print("These are the demo-day candidates. Trade only these, not the book.")
    else:
        print("\nNothing clears return>0 + PF>1.1 + positive expectancy after costs.")
        print("The entry does not survive realistic costs and a walkforward. Stop here.")
    print("=" * 72)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
