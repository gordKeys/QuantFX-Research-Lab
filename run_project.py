import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(script, extra_args=None):
    cmd = [PYTHON, str(ROOT / "scripts" / script)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd, cwd=str(ROOT))


def print_status(mode, args):
    print("\n=== QuantFX Launcher ===")
    print(f"Mode: {mode}")
    print(f"Project root: {ROOT}")
    if args.symbol:
        print(f"Symbols: {', '.join(args.symbol)}")
    if args.data:
        print(f"Data files: {', '.join(args.data)}")
    if mode == "live":
        print(f"Dry run: {args.dry_run}")
        print(f"Loop once: {args.loop_once}")
        print(f"Max consecutive losses: {args.max_consecutive_losses or 3}")
    print("======================\n")


def main():
    parser = argparse.ArgumentParser(description="QuantFX project launcher")
    parser.add_argument(
        "mode",
        choices=["test", "walkforward", "sweep", "combo", "focus", "live", "null", "nullreport", "milestone", "tournament", "export", "diagnostic", "analyze", "lossreport", "scorereport", "backtest_live_logic", "actioncounts", "entryanalyzer",
                 "export_structure", "screen", "commission", "levels", "battery", "horizon", "barrier", "simulate", "drift", "walkforward", "wfmomentum", "regime_probe", "carry_probe"],
        help="Choose what to run",
    )
    parser.add_argument("--symbol", action="append", help="Repeatable symbol filter")
    parser.add_argument("--symbols", nargs="+", help="Space-separated symbol list for export mode")
    parser.add_argument("--data", action="append", help="Repeatable CSV path filter")
    parser.add_argument("--timeframe", nargs="+", help="One or more timeframes for export/screen modes")
    parser.add_argument("--bars", type=int, help="Export bar count for export mode")
    parser.add_argument("--output-dir", help="Export output directory for export mode")
    parser.add_argument("--preset", help="export_structure mode: core | wide | a config group name")
    parser.add_argument("--swing-lookback", type=int, help="screen/levels mode: bars either side to confirm a swing")
    parser.add_argument("--barrier", type=float, help="simulate mode: barrier width in ATR")
    parser.add_argument("--risk", type=float, help="simulate mode: fraction of equity per trade")
    parser.add_argument("--equity", type=float, help="simulate mode: starting balance")
    parser.add_argument("--min-touches", type=int, help="levels mode: touches before a level counts")
    parser.add_argument("--horizon", type=int, help="levels mode: bars to look forward when judging a reaction")
    parser.add_argument("--reaction-atr", type=float, help="levels mode: move size in ATR that counts as a reaction")
    parser.add_argument("--commission-per-lot", type=float, help="screen mode: round-trip commission per lot")
    parser.add_argument("--data-dir", help="screen mode: directory holding exported CSVs")
    parser.add_argument("--days", type=int, help="Analysis window in days for analyze mode")
    parser.add_argument("--output", help="Analyzer output CSV path")
    parser.add_argument("--input", help="Input CSV path for lossreport mode")
    parser.add_argument("--logs", nargs="+", help="Log file(s) for scorereport mode")
    parser.add_argument("--trades", help="Trade CSV for scorereport mode")
    parser.add_argument("--magic-number", type=int, help="Magic number for live/null/analyze modes")
    parser.add_argument("--compare-old", action="store_true", help="backtest_live_logic mode: also run pre-tuning tiers")
    parser.add_argument("--folds", type=int, help="backtest_live_logic mode: walk-forward fold count")
    parser.add_argument("--session-hours", action="store_true", help="carry_probe: print per-hour session table")
    parser.add_argument("--giveback-scale", type=float, help="backtest_live_logic mode: test giveback buffer scaled by this factor")
    parser.add_argument("--drop", action="append", help="entryanalyzer mode: component(s) to drop for a combo test (repeatable)")
    parser.add_argument("--require-trend-alignment", action="store_true", help="entryanalyzer mode: combine with --drop")
    parser.add_argument("--combo-min-score", type=int, help="entryanalyzer mode: min_score for the --drop combo test")
    parser.add_argument("--dry-run", action="store_true", help="Live mode only")
    parser.add_argument("--loop-once", action="store_true", help="Live mode only")
    parser.add_argument("--market-open-buffer-minutes", type=int, help="Live mode only: minutes after weekly open to suppress new entries (0 disables)")
    parser.add_argument("--max-trades-per-day", type=int, help="Live mode only: hard cap on entries per rolling 24h across all symbols (0 disables)")
    parser.add_argument("--max-consecutive-losses", type=int, help="Live mode only")
    args = parser.parse_args()

    forwarded = []
    for item in args.symbol or []:
        forwarded.extend(["--symbol", item])
    for item in args.data or []:
        forwarded.extend(["--data", item])
    if args.symbols:
        forwarded.extend(["--symbols", *args.symbols])
    if args.timeframe:
        forwarded.extend(["--timeframe", *args.timeframe])
    if args.bars is not None:
        forwarded.extend(["--bars", str(args.bars)])
    if args.output_dir:
        forwarded.extend(["--output-dir", args.output_dir])
    if args.preset:
        forwarded.extend(["--preset", args.preset])
    if args.swing_lookback is not None:
        forwarded.extend(["--swing-lookback", str(args.swing_lookback)])
    if args.barrier is not None:
        forwarded.extend(["--barrier", str(args.barrier)])
    if args.risk is not None:
        forwarded.extend(["--risk", str(args.risk)])
    if args.equity is not None:
        forwarded.extend(["--equity", str(args.equity)])
    if args.min_touches is not None:
        forwarded.extend(["--min-touches", str(args.min_touches)])
    if args.horizon is not None:
        forwarded.extend(["--horizon", str(args.horizon)])
    if args.reaction_atr is not None:
        forwarded.extend(["--reaction-atr", str(args.reaction_atr)])
    if args.commission_per_lot is not None:
        forwarded.extend(["--commission-per-lot", str(args.commission_per_lot)])
    if args.data_dir:
        forwarded.extend(["--data-dir", args.data_dir])
    if args.days is not None:
        forwarded.extend(["--days", str(args.days)])
    if args.output:
        forwarded.extend(["--output", args.output])
    if args.input:
        forwarded.extend(["--input", args.input])
    if args.logs:
        forwarded.extend(["--logs", *args.logs])
    if args.trades:
        forwarded.extend(["--trades", args.trades])
    if args.magic_number is not None:
        forwarded.extend(["--magic-number", str(args.magic_number)])
    if args.compare_old:
        forwarded.append("--compare-old")
    if args.folds is not None:
        forwarded.extend(["--folds", str(args.folds)])
    if args.session_hours:
        forwarded.append("--session-hours")
    if args.giveback_scale is not None:
        forwarded.extend(["--giveback-scale", str(args.giveback_scale)])
    if args.drop:
        for component in args.drop:
            forwarded.extend(["--drop", component])
    if args.require_trend_alignment:
        forwarded.append("--require-trend-alignment")
    if args.combo_min_score is not None:
        forwarded.extend(["--combo-min-score", str(args.combo_min_score)])

    if args.mode == "test":
        print_status(args.mode, args)
        return run_script("evaluate_multi_symbol.py", forwarded)

    if args.mode == "walkforward":
        print_status(args.mode, args)
        return run_script("walkforward_multi_symbol.py", forwarded)

    if args.mode == "sweep":
        print_status(args.mode, args)
        return run_script("sweep_mean_reversion.py", forwarded)

    if args.mode == "combo":
        print_status(args.mode, args)
        return run_script("run_symbol_combo.py", forwarded)

    if args.mode == "focus":
        print_status(args.mode, args)
        return run_script("focused_combo_report.py", forwarded)

    if args.mode == "live":
        print_status(args.mode, args)
        live_args = forwarded[:]
        if args.dry_run:
            live_args.append("--dry-run")
        if args.loop_once:
            live_args.append("--loop-once")
        if args.max_consecutive_losses is not None:
            live_args.extend(["--max-consecutive-losses", str(args.max_consecutive_losses)])
        if args.market_open_buffer_minutes is not None:
            live_args.extend(["--market-open-buffer-minutes", str(args.market_open_buffer_minutes)])
        if args.max_trades_per_day is not None:
            live_args.extend(["--max-trades-per-day", str(args.max_trades_per_day)])
        if not live_args:
            live_symbols_file = ROOT / "configs" / "live_symbols.json"
            if live_symbols_file.exists():
                try:
                    with live_symbols_file.open("r", encoding="utf-8") as handle:
                        live_symbols = json.load(handle).get("symbols", [])
                    if live_symbols:
                        live_args = ["--symbols", *live_symbols]
                except Exception:
                    pass
        if not live_args:
            live_args = ["--symbols", "EURUSD", "GBPUSD"]
        if "--magic-number" not in live_args:
            live_args.extend(["--magic-number", "26072026"])
        return run_script("live_runner.py", live_args)

    if args.mode == "null":
        print_status(args.mode, args)
        null_args = forwarded[:]
        if args.dry_run:
            null_args.append("--dry-run")
        if args.loop_once:
            null_args.append("--loop-once")
        if args.max_consecutive_losses is not None:
            null_args.extend(["--max-consecutive-losses", str(args.max_consecutive_losses)])
        if not null_args:
            live_symbols_file = ROOT / "configs" / "live_symbols.json"
            if live_symbols_file.exists():
                try:
                    with live_symbols_file.open("r", encoding="utf-8") as handle:
                        live_symbols = json.load(handle).get("symbols", [])
                    if live_symbols:
                        null_args = ["--symbols", *live_symbols]
                except Exception:
                    pass
        if not null_args:
            null_args = ["--symbols", "EURUSD", "GBPUSD"]
        if "--magic-number" not in null_args:
            null_args.extend(["--magic-number", "26072027"])
        return run_script("null_trader.py", null_args)

    if args.mode == "nullreport":
        print_status(args.mode, args)
        return run_script("null_combo_report.py", forwarded)

    if args.mode == "milestone":
        print_status(args.mode, args)
        return run_script("milestone_report.py", forwarded)

    if args.mode == "tournament":
        print_status(args.mode, args)
        return run_script("tournament_report.py", forwarded)

    if args.mode == "export":
        print_status(args.mode, args)
        return run_script("export_mt5_data.py", forwarded)

    if args.mode == "export_structure":
        print_status(args.mode, args)
        return run_script("export_structure_data.py", forwarded)

    if args.mode == "regime_probe":
        print_status(args.mode, args)
        return run_script("regime_probe.py", forwarded)

    if args.mode == "carry_probe":
        print_status(args.mode, args)
        return run_script("carry_probe.py", forwarded)

    if args.mode == "wfmomentum":
        print_status(args.mode, args)
        return run_script("walkforward_momentum.py", forwarded)

    if args.mode == "drift":
        print_status(args.mode, args)
        return run_script("drift_decompose.py", forwarded)

    if args.mode == "simulate":
        print_status(args.mode, args)
        return run_script("simulate_signal.py", forwarded)

    if args.mode == "barrier":
        print_status(args.mode, args)
        return run_script("barrier_sweep.py", forwarded)

    if args.mode == "horizon":
        print_status(args.mode, args)
        return run_script("horizon_sweep.py", forwarded)

    if args.mode == "battery":
        print_status(args.mode, args)
        return run_script("hypothesis_battery.py", forwarded)

    if args.mode == "levels":
        print_status(args.mode, args)
        return run_script("level_report.py", forwarded)

    if args.mode == "commission":
        print_status(args.mode, args)
        return run_script("measure_commission.py", forwarded)

    if args.mode == "screen":
        print_status(args.mode, args)
        return run_script("instrument_screener.py", forwarded)

    if args.mode == "diagnostic":
        print_status(args.mode, args)
        return run_script("diagnostic_strategy_report.py", forwarded)

    if args.mode == "analyze":
        print_status(args.mode, args)
        return run_script("trade_analyzer.py", forwarded)

    if args.mode == "lossreport":
        print_status(args.mode, args)
        return run_script("loss_diagnostics.py", forwarded)

    if args.mode == "scorereport":
        print_status(args.mode, args)
        return run_script("score_quality_report.py", forwarded)

    if args.mode == "actioncounts":
        print_status(args.mode, args)
        return run_script("action_counts.py", forwarded)

    if args.mode == "entryanalyzer":
        print_status(args.mode, args)
        return run_script("entry_quality_analyzer.py", forwarded)

    if args.mode == "backtest_live_logic":
        print_status(args.mode, args)
        return run_script("backtest_live_logic.py", forwarded)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
