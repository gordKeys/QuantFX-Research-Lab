"""
Sequential backtest of the dip-entry signal -- the honest test.

Everything so far has been a barrier statistic: for each event independently,
did price reach +k ATR before -k ATR? That abstraction assumes you take every
signal, with perfect sizing, no overlap, no capital constraint, and a fill at a
price you cannot actually trade at.

A 58-61% win rate at symmetric barriers survived six checks:

  1. shifted-zone control    zone location is irrelevant, but direction is not
  2. tie handling            ties are 0.5-2% of events; result stable either way
  3. direction flip          edge(+d) and edge(-d) are mirror images, so the
                             machinery is not manufacturing edge
  4. resolution rates        events and controls resolve at the same rate (77%)
  5. forced 50/50 control    control sits at 0.50, so the null is sound
  6. entry at next bar open  survives; not a bar-boundary artifact

I still cannot explain why it is that large, and an unexplained 60% at 1:1 is a
reason for caution, not confidence. So this script stops asking whether the
statistic is significant and asks whether the money is real:

  - one position at a time per instrument, so overlapping signals are skipped
    the way they would be in live trading
  - entry at the NEXT bar's open, plus half the spread, because you cannot
    trade at a bar's closing tick
  - exit pays the spread again, plus measured commission
  - fixed-fractional sizing off a real account balance
  - equity curve, max drawdown, and trades per day
  - a train/test split: the last 30% of history is never used to decide
    anything, it is only scored

If the edge is real it survives all of that with a smaller number. If it was an
artifact of the abstraction, this is where it dies.

    python run_project.py simulate --timeframe H4 --barrier 3.0
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.hypotheses import _find_fvgs, _zone_events
from analysis.structure import atr

CLASS_COMMISSION = {"fx": 5.04, "metal": 6.00, "index": 0.0, "energy": 0.0, "crypto": 0.0}


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
    """point, spread in price, $ per price-unit per lot, commission per lot."""
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

    commission_map = Path("configs/commission_map.json")
    per_lot = None
    if commission_map.exists():
        payload = json.loads(commission_map.read_text())
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


def simulate(df, spec, events, barrier, risk_frac, start_equity, split_at):
    """
    Walk forward one bar at a time. Returns per-trade records.

    Position sizing: risk `risk_frac` of current equity across the stop
    distance. If the instrument has no dollar conversion in its metadata we
    skip it rather than guess -- a wrong contract size silently invents or
    destroys returns.
    """
    if not spec["dollars_per_unit"]:
        return None

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    atr_values = atr(df).to_numpy()
    spread_price = df["spread"].to_numpy() * spec["point"]

    event_at = {}
    for i, direction in events:
        event_at.setdefault(i, direction)

    equity = start_equity
    trades = []
    position = None

    for i in range(len(df) - 1):
        # ---- manage an open position ----
        if position is not None:
            direction = position["dir"]
            hit_tp = high[i] >= position["tp"] if direction > 0 else low[i] <= position["tp"]
            hit_sl = low[i] <= position["sl"] if direction > 0 else high[i] >= position["sl"]

            exit_price = None
            if hit_tp and hit_sl:
                exit_price = position["sl"]      # assume the worst intrabar order
            elif hit_tp:
                exit_price = position["tp"]
            elif hit_sl:
                exit_price = position["sl"]

            if exit_price is not None:
                # pay the spread on the way out too
                exit_fill = exit_price - direction * spread_price[i] / 2
                move = (exit_fill - position["entry"]) * direction
                gross = move * spec["dollars_per_unit"] * position["lots"]
                commission = spec["commission_per_lot"] * position["lots"]
                pnl = gross - commission
                equity += pnl
                trades.append({
                    "entry_bar": position["bar"],
                    "exit_bar": i,
                    "dir": direction,
                    "lots": position["lots"],
                    "pnl": pnl,
                    "commission": commission,
                    "equity": equity,
                    "in_sample": position["bar"] < split_at,
                })
                position = None

        # ---- open a new position ----
        if position is None and i in event_at and equity > 0:
            band = atr_values[i]
            if np.isnan(band) or band <= 0:
                continue
            direction = event_at[i]
            entry = open_[i + 1] + direction * spread_price[i] / 2
            stop_distance = band * barrier
            risk_dollars = equity * risk_frac
            lots = risk_dollars / (stop_distance * spec["dollars_per_unit"])
            step = spec["volume_step"]
            lots = max(spec["volume_min"], np.floor(lots / step) * step)
            if lots <= 0:
                continue
            position = {
                "bar": i + 1,
                "dir": direction,
                "entry": entry,
                "sl": entry - direction * stop_distance,
                "tp": entry + direction * stop_distance,
                "lots": lots,
            }

    return trades


def summarise(trades, label, start_equity, days):
    if not trades:
        return None
    pnl = np.array([t["pnl"] for t in trades])
    equity = start_equity + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    wins = (pnl > 0).sum()
    return {
        "period": label,
        "trades": len(trades),
        "per_day": len(trades) / max(days, 1),
        "win_rate": wins / len(trades),
        "total_pnl": pnl.sum(),
        "return_pct": pnl.sum() / start_equity * 100,
        "max_dd_pct": drawdown.max() * 100 if len(drawdown) else 0.0,
        "commission": sum(t["commission"] for t in trades),
    }


def main():
    parser = argparse.ArgumentParser(description="Sequential backtest with real costs.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--barrier", type=float, default=3.0)
    parser.add_argument("--risk", type=float, default=0.005, help="fraction of equity per trade")
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--output", default="reports/simulation.csv")
    args = parser.parse_args()

    pattern = re.compile(rf"^(.+)_{re.escape(args.timeframe)}\.csv$")
    found = [
        (pattern.match(p.name).group(1), p)
        for p in sorted(Path(args.data_dir).glob(f"*_{args.timeframe}.csv"))
        if pattern.match(p.name)
    ]
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    print(f"\n=== Sequential backtest -- {args.timeframe}, {args.barrier} ATR barriers, "
          f"{args.risk*100:.1f}% risk/trade ===")
    print("One position at a time. Entry at next bar open + half spread. Exit pays")
    print("spread again plus measured commission. Last 30% of history is out-of-sample.\n")
    print(f"{'symbol':10}{'period':>8}{'trades':>8}{'/day':>7}{'win%':>7}"
          f"{'return%':>10}{'maxDD%':>9}{'fees$':>10}")

    rows = []
    for symbol, path in found:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 2000:
            continue

        spec = instrument_spec(args.data_dir, symbol, args.timeframe, df)
        atr_values = atr(df).to_numpy()
        events = _zone_events(df, atr_values, _find_fvgs(df, atr_values))
        if len(events) < 50:
            continue

        split_at = int(len(df) * 0.7)
        trades = simulate(df, spec, events, args.barrier, args.risk, args.equity, split_at)
        if trades is None:
            print(f"{symbol:10}   skipped -- no tick value in meta, cannot size positions")
            continue
        if not trades:
            continue

        span = (df.index.max() - df.index.min()).days
        for label, subset in [
            ("in", [t for t in trades if t["in_sample"]]),
            ("OUT", [t for t in trades if not t["in_sample"]]),
        ]:
            days = span * (0.7 if label == "in" else 0.3)
            stats = summarise(subset, label, args.equity, days)
            if not stats:
                continue
            stats["symbol"] = symbol
            rows.append(stats)
            print(f"{symbol:10}{label:>8}{stats['trades']:>8}{stats['per_day']:>7.2f}"
                  f"{stats['win_rate']*100:>7.1f}{stats['return_pct']:>10.2f}"
                  f"{stats['max_dd_pct']:>9.2f}{stats['commission']:>10.0f}")

    if not rows:
        raise SystemExit("\nNo instrument produced tradeable output.")

    table = pd.DataFrame(rows)
    out_sample = table[table["period"] == "OUT"]
    in_sample = table[table["period"] == "in"]

    print("\n" + "=" * 70)
    print(f"IN-SAMPLE : {len(in_sample)} instruments, "
          f"{(in_sample['return_pct'] > 0).sum()} profitable, "
          f"median return {in_sample['return_pct'].median():+.2f}%, "
          f"median maxDD {in_sample['max_dd_pct'].median():.2f}%")
    print(f"OUT-SAMPLE: {len(out_sample)} instruments, "
          f"{(out_sample['return_pct'] > 0).sum()} profitable, "
          f"median return {out_sample['return_pct'].median():+.2f}%, "
          f"median maxDD {out_sample['max_dd_pct'].median():.2f}%")
    print(f"Total trades/day across the book: "
          f"{out_sample['per_day'].sum():.1f}")
    print("=" * 70)
    print("\nThe out-of-sample row is the only one that means anything. If it is")
    print("materially worse than in-sample, the signal was fitted. If both are")
    print("positive with a drawdown you could actually sit through, that is the")
    print("first thing in this project worth forward-testing on demo.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
