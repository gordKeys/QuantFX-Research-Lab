"""
Loss attribution diagnostics.

Reads the CSV produced by trade_analyzer.py (logs/trade_analysis.csv by
default) and answers two questions that trade_analyzer's per-symbol table
doesn't directly answer:

1. ENTRY vs EXIT: how much of total PnL comes from trades that never even
   reached profitability (bad entry / bad timing) vs trades that reached
   profit and then gave it back (exit management) vs trades that reached
   profit and kept some of it. This matters because a profit-retention
   fix (trailing stops, giveback close, etc.) can only ever help the
   middle bucket -- if most of the damage is in the first bucket, no
   amount of exit tuning will fix it; the entry signal itself needs work.

2. HOUR OF DAY: PnL, win rate, and trade count bucketed by UTC open hour,
   with a low-sample warning so a couple of unlucky/lucky hours in a
   1-2 day sample don't get mistaken for a real session effect.

Usage:
    python scripts/loss_diagnostics.py --input logs/trade_analysis.csv
"""
from bootstrap import add_project_root
add_project_root()

import argparse
from pathlib import Path

import pandas as pd


MIN_SAMPLE_FOR_CONFIDENCE = 15


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("open_time", "close_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def entry_vs_exit_report(df: pd.DataFrame) -> None:
    print("\n=== ENTRY QUALITY vs EXIT QUALITY ===")
    total_pnl = df["profit_usd"].sum()
    never_profit = df[df["mfe_usd"] <= 0]
    peaked = df[df["mfe_usd"] > 0]
    gave_back_to_loss = peaked[peaked["profit_usd"] <= 0]
    kept_profit = peaked[peaked["profit_usd"] > 0]

    def pct_of_total(x):
        return (x / total_pnl * 100.0) if total_pnl else 0.0

    print(f"Total PnL: {total_pnl:.2f} across {len(df)} trades")
    print(
        f"Never reached profit (bad entry, mfe<=0): {len(never_profit)} trades | "
        f"PnL {never_profit['profit_usd'].sum():.2f} "
        f"({pct_of_total(never_profit['profit_usd'].sum()):.0f}% of total loss/gain)"
    )
    print(
        f"Reached profit then gave it all back (exit issue): {len(gave_back_to_loss)} trades | "
        f"PnL {gave_back_to_loss['profit_usd'].sum():.2f}"
    )
    print(
        f"Reached profit and kept some/all of it: {len(kept_profit)} trades | "
        f"PnL {kept_profit['profit_usd'].sum():.2f}"
    )

    never_share = abs(never_profit["profit_usd"].sum())
    exit_share = abs(gave_back_to_loss["profit_usd"].sum()) - kept_profit["profit_usd"].sum()
    if never_share + max(exit_share, 0) > 0:
        never_pct = never_share / (never_share + max(exit_share, 0)) * 100.0
        print(
            f"\n-> Of the recoverable damage, roughly {never_pct:.0f}% is entry-side "
            f"(trades that were simply wrong) vs {100 - never_pct:.0f}% net exit-side "
            f"(giveback minus what's already retained)."
        )
        if never_pct >= 60:
            print(
                "   This says exit/profit-retention tuning has limited upside here -- "
                "the bigger lever is entry signal quality (fewer/ better-filtered trades), "
                "not tighter trailing stops."
            )


def hour_of_day_report(df: pd.DataFrame) -> None:
    if "open_time" not in df.columns:
        print("\nNo open_time column found; skipping hour-of-day report.")
        return

    print("\n=== HOUR OF DAY (UTC) ===")
    work = df.copy()
    work["hour"] = work["open_time"].dt.hour
    grouped = work.groupby("hour").agg(
        trades=("profit_usd", "count"),
        total_pnl=("profit_usd", "sum"),
        avg_pnl=("profit_usd", "mean"),
        win_rate=("profit_usd", lambda s: (s > 0).mean()),
    )
    print(f"{'hour':>4} | {'trades':>6} | {'total_pnl':>10} | {'avg_pnl':>8} | {'win_rate':>8} | note")
    print("-" * 60)
    for hour, row in grouped.sort_index().iterrows():
        note = "" if row["trades"] >= MIN_SAMPLE_FOR_CONFIDENCE else "low sample, don't act on this yet"
        print(
            f"{hour:4d} | {int(row['trades']):6d} | {row['total_pnl']:10.2f} | "
            f"{row['avg_pnl']:8.2f} | {row['win_rate']:8.2%} | {note}"
        )

    low_sample_hours = int((grouped["trades"] < MIN_SAMPLE_FOR_CONFIDENCE).sum())
    if low_sample_hours:
        print(
            f"\n{low_sample_hours}/{len(grouped)} hour buckets have fewer than "
            f"{MIN_SAMPLE_FOR_CONFIDENCE} trades. Treat any single-hour result there as "
            "a hint, not a rule -- keep logging and re-run this report as more live "
            "trades accumulate before hard-coding an hour blocklist."
        )


def symbol_by_hour(df: pd.DataFrame) -> None:
    if "open_time" not in df.columns:
        return
    print("\n=== SYMBOL x HOUR TOTAL PNL (trades in parens) ===")
    work = df.copy()
    work["hour"] = work["open_time"].dt.hour
    pivot_pnl = work.pivot_table(index="symbol", columns="hour", values="profit_usd", aggfunc="sum")
    pivot_count = work.pivot_table(index="symbol", columns="hour", values="profit_usd", aggfunc="count")
    hours = sorted(pivot_pnl.columns)
    header = "symbol  | " + " | ".join(f"{h:>10d}" for h in hours)
    print(header)
    for symbol in pivot_pnl.index:
        cells = []
        for h in hours:
            pnl = pivot_pnl.loc[symbol, h] if h in pivot_pnl.columns else None
            cnt = pivot_count.loc[symbol, h] if h in pivot_count.columns else None
            if pd.isna(pnl):
                cells.append(f"{'--':>10}")
            else:
                cells.append(f"{pnl:6.1f}({int(cnt)})")
        print(f"{symbol:7} | " + " | ".join(f"{c:>10}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description="Attribute PnL to entry quality vs exit quality vs hour of day")
    parser.add_argument("--input", default="logs/trade_analysis.csv")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"No such file: {path}. Run trade_analyzer.py first to produce it.")

    df = _load(path)
    if df.empty:
        print("Input CSV has no rows.")
        return

    entry_vs_exit_report(df)
    hour_of_day_report(df)
    symbol_by_hour(df)


if __name__ == "__main__":
    main()
