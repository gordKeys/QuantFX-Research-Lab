"""
Barrier sweep -- does the residual edge survive wider targets, after costs?

What the battery actually found, once the SMC labels are stripped off:

  Order blocks, shifted order blocks, fair value gaps and shifted gaps all
  score the same (+0.03 to +0.07, 5-16 sigma). Moving a zone 1.5 ATR to a
  meaningless price does not weaken it -- for order blocks the SHIFTED version
  scored higher on both timeframes. So the zone is not the cause.

  What every one of those tests has in common is the bet: a bullish zone sits
  below price, price falls into it, and we buy. A bearish zone sits above,
  price rises into it, and we sell. Strip the vocabulary and it is "fade the
  approach" -- short-horizon mean reversion. ctl_low_close measures the same
  thing more weakly (-0.015 both timeframes).

  So: one effect, measured five ways, none of which required an order block or
  a fair value gap to exist.

Why it has stayed invisible: at 1 ATR barriers on EURUSD M15, a 6.2-point edge
is worth about 0.68 pips gross against roughly 0.7 pips of spread plus
commission. It nets to zero. That is not a coincidence -- an edge that cleared
costs at this frequency would have been arbitraged away.

The one question left that matters for "few substantial trades": mean reversion
might decay as the target widens, or it might hold. Cost is FIXED per trade, so
if the edge holds even partially at 2-4 ATR, the cost ratio collapses and the
trade becomes viable. If the edge decays in proportion, it never clears costs
at any width and the project is finished.

This script measures exactly that, net of your measured per-instrument costs.

    python run_project.py barrier --timeframe M15
    python run_project.py barrier --timeframe H4
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.hypotheses import _control_events, _find_fvgs, _rate, _zone_events
from analysis.structure import atr

BARRIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
COMMISSION_MAP = Path("configs/commission_map.json")

CLASS_COMMISSION = {"fx": 5.04, "metal": 6.00, "index": 0.0, "energy": 0.0, "crypto": 0.0}


def classify(symbol):
    u = symbol.upper()
    if any(h in u for h in ("BTC", "ETH")):
        return "crypto"
    if any(h in u for h in ("XAU", "XAG", "GOLD", "SILVER")):
        return "metal"
    if "OIL" in u:
        return "energy"
    if any(u.startswith(h) for h in ("US30", "US500", "NAS100", "GER40", "US100")):
        return "index"
    return "fx"


def load_costs(data_dir, symbol, timeframe, df):
    """Round-trip cost in PRICE terms for this instrument."""
    meta_path = Path(data_dir) / f"{symbol}_{timeframe}_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    point = float(meta.get("point") or 0)
    if not point:
        sample = df["close"].dropna().head(200)
        decimals = max(
            (len(f"{v:.10f}".rstrip('0').split('.')[1]) for v in sample), default=5
        )
        point = 10.0 ** (-min(decimals, 5))

    spread_price = float(df["spread"].median()) * point

    per_lot = None
    if COMMISSION_MAP.exists():
        payload = json.loads(COMMISSION_MAP.read_text())
        entry = payload.get("symbols", {}).get(symbol)
        if entry:
            per_lot = entry.get("per_lot_round_trip")
    if per_lot is None:
        per_lot = CLASS_COMMISSION.get(classify(symbol), 0.0)

    commission_price = 0.0
    tick_value = meta.get("trade_tick_value")
    tick_size = meta.get("trade_tick_size") or point
    if tick_value and tick_size and per_lot:
        dollars_per_price_unit = float(tick_value) / float(tick_size)
        if dollars_per_price_unit > 0:
            commission_price = per_lot / dollars_per_price_unit

    return spread_price + commission_price


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [
        (pattern.match(p.name).group(1), p)
        for p in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
        if pattern.match(p.name)
    ]


def main():
    parser = argparse.ArgumentParser(description="Sweep barrier width, net of costs.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--output", default="reports/barrier_sweep.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv files in {args.data_dir}/.")

    prepared = {}
    for symbol, path in found:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 2000:
            continue
        atr_values = atr(df).to_numpy()
        prepared[symbol] = {
            "df": df,
            "atr": atr_values,
            "median_atr": float(np.nanmedian(atr_values)),
            "cost": load_costs(args.data_dir, symbol, args.timeframe, df),
            "events": _zone_events(df, atr_values, _find_fvgs(df, atr_values)),
        }

    print(f"\n=== Barrier sweep -- {args.timeframe}, {len(prepared)} instruments, "
          f"{args.horizon} bar horizon ===")
    print("Zone-entry (fade the approach) at widening targets, net of measured cost.\n")
    print(f"{'barrier':>9}{'edge':>10}{'sigma':>8}{'events':>9}"
          f"{'gross/trade':>13}{'cost':>10}{'net':>10}{'net (ATR)':>11}")

    rows = []
    for k in BARRIERS:
        wins = events = c_wins = c_events = 0
        gross_sum = cost_sum = weight = 0.0

        for symbol, blob in prepared.items():
            df, atr_values = blob["df"], blob["atr"]
            if len(blob["events"]) < 30:
                continue
            high = df["high"].to_numpy(); low = df["low"].to_numpy()
            close = df["close"].to_numpy()

            w, t = _rate(blob["events"], high, low, close, atr_values, args.horizon, k)
            dirs = [d for _, d in blob["events"]]
            ctl = _control_events(len(df), atr_values, dirs, args.horizon, 7)
            cw, ct = _rate(ctl, high, low, close, atr_values, args.horizon, k)
            if t < 30 or ct < 30:
                continue

            wins += w; events += t; c_wins += cw; c_events += ct
            # per-instrument economics, weighted by event count
            sym_edge = w / t - cw / ct
            gross_sum += (2 * sym_edge * k * blob["median_atr"]) * t
            cost_sum += blob["cost"] * t
            weight += t

        if not events or not weight:
            continue

        rate = wins / events
        c_rate = c_wins / c_events
        err = float(np.sqrt(rate * (1 - rate) / events + c_rate * (1 - c_rate) / c_events))
        edge = rate - c_rate
        gross = gross_sum / weight
        cost = cost_sum / weight
        net = gross - cost
        # express net in ATR units so instruments are comparable
        mean_atr = np.mean([b["median_atr"] for b in prepared.values()])

        print(f"{k:>9.1f}{edge:>+10.4f}{edge/err:>+8.1f}{events:>9}"
              f"{gross:>13.6f}{cost:>10.6f}{net:>+10.6f}{net/mean_atr:>+11.3f}")
        rows.append({"timeframe": args.timeframe, "barrier_atr": k, "edge": edge,
                     "sigma": edge / err, "events": events, "gross": gross,
                     "cost": cost, "net": net})

    print("\ngross/trade = 2 x edge x barrier x ATR, averaged across instruments")
    print("cost        = median spread + measured commission, per round trip")
    print("net         = what you would actually keep, per trade, in price terms")
    print("\nIf `edge` holds roughly flat as barrier widens, net turns positive and")
    print("this is tradeable at low frequency. If `edge` shrinks in proportion to")
    print("the barrier, net stays at or below zero at every width and it is not.")

    if rows:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        if out.exists():
            table = pd.concat([pd.read_csv(out), table], ignore_index=True)
            table = table.drop_duplicates(subset=["timeframe", "barrier_atr"], keep="last")
        table.to_csv(out, index=False)
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
