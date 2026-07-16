from bootstrap import add_project_root
add_project_root()

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional chart support
    plt = None

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
    gave_back_usd: float
    captured_vs_peak: float
    peak_to_final_pct: float
    positive_peak_then_loss: bool
    closed_in_profit: bool


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


def _generate_charts(df: pd.DataFrame, grouped: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        print("Charts skipped: matplotlib unavailable.")
        return

    chart_dir = output_path.parent
    chart_dir.mkdir(parents=True, exist_ok=True)

    symbol_colors = {symbol: color for symbol, color in zip(grouped.index.tolist(), ["#2e86de", "#16a085", "#e67e22", "#c0392b", "#8e44ad", "#27ae60", "#d35400", "#2980b9"])}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trade Analyzer Diagnostic Report", fontsize=14)

    axes[0, 0].scatter(df["mfe_usd"], df["profit_usd"], alpha=0.65, c=[symbol_colors.get(sym, "#555555") for sym in df["symbol"]])
    axes[0, 0].axhline(0, color="#666666", linewidth=0.8)
    axes[0, 0].axvline(0, color="#666666", linewidth=0.8)
    axes[0, 0].set_title("MFE vs Final PnL")
    axes[0, 0].set_xlabel("Peak favorable excursion ($)")
    axes[0, 0].set_ylabel("Final PnL ($)")

    retention = grouped["avg_retained_peak"] if "avg_retained_peak" in grouped.columns else pd.Series(dtype=float)
    if not retention.empty:
        axes[0, 1].bar(retention.index, retention.values, color=[symbol_colors.get(sym, "#2e86de") for sym in retention.index])
        axes[0, 1].set_title("Average Retained Peak by Symbol")
        axes[0, 1].set_ylabel("Retained peak %")
        axes[0, 1].tick_params(axis='x', rotation=25)

    axes[1, 0].hist(df["gave_back_usd"].dropna(), bins=12, color="#5dade2", edgecolor="#1b4f72")
    axes[1, 0].set_title("Giveback Distribution")
    axes[1, 0].set_xlabel("Gave back from peak ($)")
    axes[1, 0].set_ylabel("Trades")

    per_symbol = df.groupby("symbol")["positive_peak_then_loss"].mean().sort_values(ascending=False) if "positive_peak_then_loss" in df.columns else pd.Series(dtype=float)
    if not per_symbol.empty:
        axes[1, 1].bar(per_symbol.index, per_symbol.values, color="#c0392b")
        axes[1, 1].set_title("Rate of Positive-Peak -> Loss Trades")
        axes[1, 1].set_ylabel("Share of trades")
        axes[1, 1].tick_params(axis='x', rotation=25)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    png_path = chart_dir / f"{output_path.stem}_charts.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Saved chart: {png_path}")


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
            mfe_usd, mae_usd = _price_favorability(broker, symbol, volume, direction, open_price=entry_price, close_price=exit_price, bars=bars)
            gave_back_usd = max(mfe_usd - profit_usd, 0.0)
            captured_vs_peak = (profit_usd / mfe_usd) if mfe_usd not in (0, 0.0) else pd.NA
            peak_to_final_pct = ((profit_usd / mfe_usd) * 100.0) if mfe_usd not in (0, 0.0) else pd.NA
            positive_peak_then_loss = bool(mfe_usd > 0 and profit_usd <= 0)
            closed_in_profit = bool(profit_usd > 0)
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
                    gave_back_usd=gave_back_usd,
                    captured_vs_peak=captured_vs_peak,
                    peak_to_final_pct=peak_to_final_pct,
                    positive_peak_then_loss=positive_peak_then_loss,
                    closed_in_profit=closed_in_profit,
                )
            )

        if not rows:
            print("No closed trades found for analysis.")
            return

        df = pd.DataFrame([row.__dict__ for row in rows]).sort_values(["close_time", "symbol"])
        df["duration_min"] = (df["close_time"] - df["open_time"]).dt.total_seconds() / 60.0
        df["r_multiple"] = df["profit_usd"] / df["mae_usd"].abs().replace(0, np.nan)
        df["gave_back"] = df["gave_back_usd"]
        df["captured_vs_peak"] = pd.to_numeric(df["captured_vs_peak"], errors="coerce")
        df["retained_peak_pct"] = df["captured_vs_peak"] * 100.0
        df["positive_peak_then_loss"] = df["positive_peak_then_loss"].astype(bool)
        df["peak_hit"] = df["mfe_usd"] > 0
        df["peak_to_loss"] = df["peak_hit"] & (df["profit_usd"] <= 0)
        # NOTE: mae_usd is 0 for break-even/no-adverse-move trades. Using np.nan (a real
        # float) here -- instead of the previous pd.NA -- keeps this column a normal
        # float64 dtype so it can flow straight into .mean()/.astype(float) below without
        # pandas falling back to a mixed object dtype (which is what caused the
        # "TypeError: float() argument must be ... not 'NAType'" crash).
        df["profit_to_mae"] = df["profit_usd"] / df["mae_usd"].abs().replace(0, np.nan)
        df["mfe_minus_mae"] = df["mfe_usd"] + df["mae_usd"].abs()

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
        print(f"Positive-peak then loss trades: {df['positive_peak_then_loss'].sum()} / {len(df)}")
        print(f"Peak -> final retention rate: {df['retained_peak_pct'].mean():.2f}%")
        print(f"Trades with >50% giveback: {(df['gave_back'] > (df['mfe_usd'].abs() * 0.5)).mean():.2%}")
        print(f"Trades with peak then loss: {df['peak_to_loss'].mean():.2%}")
        print(f"Avg profit / MAE ratio: {pd.to_numeric(df['profit_to_mae'], errors='coerce').mean():.2f}")

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
                avg_retained_peak=("captured_vs_peak", lambda series: pd.to_numeric(series, errors="coerce").mean() * 100.0),
                peak_to_final_ratio=("captured_vs_peak", lambda series: pd.to_numeric(series, errors="coerce").median()),
                peak_then_loss_rate=("positive_peak_then_loss", lambda series: series.mean()),
                peak_hit_rate=("peak_hit", lambda series: series.mean()),
                loss_after_peak_rate=("peak_to_loss", lambda series: series.mean()),
                avg_profit_to_mae=("profit_to_mae", lambda series: pd.to_numeric(series, errors="coerce").mean()),
                avg_mfe_minus_mae=("mfe_minus_mae", "mean"),
                worst_loss=("profit_usd", "min"),
            )
            .sort_values(["total_pnl", "win_rate"], ascending=[False, False])
        )
        print(
            f"{'symbol':>8} | {'trades':>6} | {'win_rate':>8} | {'total_pnl':>10} | {'avg_mfe':>8} | {'avg_mae':>8} | {'giveback':>9} | {'ret%':>7} | {'peak_loss':>8} | {'pkHit':>6} | {'lossPk':>7}"
        )
        print("-" * 132)
        for symbol, row in grouped.iterrows():
            print(
                f"{symbol:>8} | {int(row['trades']):6d} | {row['win_rate']:8.2%} | {row['total_pnl']:10.2f} | {row['avg_mfe']:8.2f} | {row['avg_mae']:8.2f} | {row['avg_giveback']:9.2f} | {row['avg_retained_peak']:7.2f} | {row['peak_then_loss_rate']:8.2%} | {row['peak_hit_rate']:6.2%} | {row['loss_after_peak_rate']:7.2%}"
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved CSV: {output_path}")
        _generate_charts(df, grouped, output_path)

    broker.shutdown()


if __name__ == "__main__":
    main()
