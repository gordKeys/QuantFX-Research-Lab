from bootstrap import add_project_root
add_project_root()

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError
from timing_utils import timed


UTC = timezone.utc


@dataclass
class TradeRow:
    symbol: str
    position_id: int
    direction: str
    open_time: datetime
    close_time: datetime
    volume: float
    entry_price: float
    exit_price: float
    profit_usd: float
    mfe_usd: float
    mae_usd: float


def _ensure_utc(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if getattr(value, "tzinfo", None) is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _price_favorability(broker, symbol, volume, direction, open_price, close_price, bars):
    if bars is None or bars.empty:
        return 0.0, 0.0

    entry_type = broker.mt5.ORDER_TYPE_BUY if direction == "BUY" else broker.mt5.ORDER_TYPE_SELL
    highs = bars["high"].astype(float).tolist()
    lows = bars["low"].astype(float).tolist()

    if direction == "BUY":
        mfe_price = max(highs) if highs else open_price
        mae_price = min(lows) if lows else open_price
    else:
        mfe_price = min(lows) if lows else open_price
        mae_price = max(highs) if highs else open_price

    mfe = broker.mt5.order_calc_profit(entry_type, symbol, volume, open_price, mfe_price)
    mae = broker.mt5.order_calc_profit(entry_type, symbol, volume, open_price, mae_price)
    mfe = float(mfe or 0.0)
    mae = float(mae or 0.0)
    return mfe, mae


def _fetch_m5_bars(broker, symbol, start_time, end_time):
    bars = broker.mt5.copy_rates_range(
        symbol,
        broker.mt5.TIMEFRAME_M5,
        start_time,
        end_time + timedelta(minutes=5),
    )
    if bars is None or len(bars) == 0:
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.set_index("time")


def _pair_deals(deals):
    entries = {}
    rows = []
    for deal in sorted(deals, key=lambda item: getattr(item, "time", 0)):
        entry_kind = getattr(deal, "entry", None)
        position_id = int(getattr(deal, "position_id", 0) or getattr(deal, "position", 0) or 0)
        if position_id == 0:
            continue
        if entry_kind == 0:
            entries[position_id] = deal
            continue
        if entry_kind == 1:
            open_deal = entries.get(position_id)
            if open_deal is None:
                continue
            rows.append((open_deal, deal))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Analyze live MT5 trade quality")
    parser.add_argument("--magic-number", type=int, default=26072026)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--output", default="logs/trade_analysis.csv")
    args = parser.parse_args()

    try:
        broker = MT5BrokerAdapter(magic_number=args.magic_number)
        broker.initialize()
    except MT5UnavailableError as exc:
        raise SystemExit(f"MT5 unavailable: {exc}")

    since = datetime.now(UTC) - timedelta(days=args.days)
    with timed("trade_analysis", style="days"):
        deals = broker.history_deals_since(since, magic=args.magic_number)
        if args.symbol:
            wanted = {item.upper() for item in args.symbol}
            deals = [deal for deal in deals if getattr(deal, "symbol", "").upper() in wanted]

        closed_pairs = _pair_deals(deals)
        rows = []
        for open_deal, close_deal in closed_pairs:
            symbol = getattr(open_deal, "symbol", "")
            open_time = _ensure_utc(getattr(open_deal, "time", None))
            close_time = _ensure_utc(getattr(close_deal, "time", None))
            if open_time is None or close_time is None:
                continue

            direction = "BUY" if getattr(open_deal, "type", 0) == 0 else "SELL"
            volume = float(getattr(open_deal, "volume", 0.0) or 0.0)
            entry_price = float(getattr(open_deal, "price", 0.0) or 0.0)
            exit_price = float(getattr(close_deal, "price", 0.0) or 0.0)
            profit_usd = float(getattr(close_deal, "profit", 0.0) or 0.0)
            bars = _fetch_m5_bars(broker, symbol, open_time, close_time)
            mfe_usd, mae_usd = _price_favorability(broker, symbol, volume, direction, entry_price, exit_price, bars)
            rows.append(
                TradeRow(
                    symbol=symbol,
                    position_id=int(getattr(open_deal, "position_id", 0) or getattr(open_deal, "position", 0) or 0),
                    direction=direction,
                    open_time=open_time,
                    close_time=close_time,
                    volume=volume,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    profit_usd=profit_usd,
                    mfe_usd=mfe_usd,
                    mae_usd=mae_usd,
                )
            )

        if not rows:
            print("No closed trades found for analysis.")
            return

        df = pd.DataFrame([row.__dict__ for row in rows]).sort_values(["close_time", "symbol"])
        df["duration_min"] = (df["close_time"] - df["open_time"]).dt.total_seconds() / 60.0
        df["r_multiple"] = df["profit_usd"] / df["mae_usd"].abs().replace(0, pd.NA)
        df["gave_back"] = df["mfe_usd"] - df["profit_usd"]

        print("\n=== TRADE ANALYSIS ===")
        print(
            f"{'symbol':>8} | {'dir':>3} | {'open_time':>19} | {'close_time':>19} | "
            f"{'dur_m':>6} | {'entry':>9} | {'exit':>9} | {'pnl':>8} | {'mfe':>8} | {'mae':>8}"
        )
        print("-" * 122)
        for _, row in df.iterrows():
            print(
                f"{row['symbol']:>8} | {row['direction'][:3]:>3} | {str(row['open_time']):>19} | {str(row['close_time']):>19} | "
                f"{row['duration_min']:6.1f} | {row['entry_price']:9.5f} | {row['exit_price']:9.5f} | "
                f"{row['profit_usd']:8.2f} | {row['mfe_usd']:8.2f} | {row['mae_usd']:8.2f}"
            )

        print("\n=== SUMMARY ===")
        print(f"Trades analyzed: {len(df)}")
        print(f"Total PnL: {df['profit_usd'].sum():.2f}")
        print(f"Win rate: {(df['profit_usd'] > 0).mean():.2%}")
        print(f"Avg max favorable excursion: {df['mfe_usd'].mean():.2f}")
        print(f"Avg max adverse excursion: {df['mae_usd'].mean():.2f}")
        print(f"Avg time in trade (min): {df['duration_min'].mean():.1f}")
        print(f"Avg giveback from peak: {df['gave_back'].mean():.2f}")

        print("\n=== BY SYMBOL ===")
        grouped = (
            df.groupby("symbol")
            .agg(
                trades=("profit_usd", "count"),
                win_rate=("profit_usd", lambda series: (series > 0).mean()),
                total_pnl=("profit_usd", "sum"),
                avg_mfe=("mfe_usd", "mean"),
                avg_mae=("mae_usd", "mean"),
                avg_giveback=("gave_back", "mean"),
                worst_loss=("profit_usd", "min"),
            )
            .sort_values(["total_pnl", "win_rate"], ascending=[False, False])
        )
        print(
            f"{'symbol':>8} | {'trades':>6} | {'win_rate':>8} | {'total_pnl':>10} | {'avg_mfe':>8} | {'avg_mae':>8} | {'giveback':>9} | {'worst':>8}"
        )
        print("-" * 92)
        for symbol, row in grouped.iterrows():
            print(
                f"{symbol:>8} | {int(row['trades']):6d} | {row['win_rate']:8.2%} | {row['total_pnl']:10.2f} | {row['avg_mfe']:8.2f} | {row['avg_mae']:8.2f} | {row['avg_giveback']:9.2f} | {row['worst_loss']:8.2f}"
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved CSV: {output_path}")

    broker.shutdown()


if __name__ == "__main__":
    main()
