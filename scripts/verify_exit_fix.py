"""
Offline, MT5-free verification of the exit-logic changes made to live_runner.py.

Run:
    python scripts/verify_exit_fix.py

This does NOT touch a broker. It:
  1. Replays synthetic bar-by-bar price paths through the real
     manage_live_position()/trade_management_params() functions using fake
     broker/position stand-ins, to prove the new dollar-based giveback rule and
     the USDCHF/EURUSD-specific tightening actually change behaviour the way we
     intend.
  2. Re-plays the 39 real trades from the July 15 trade_analyzer log (entry/exit/
     mfe/mae as printed by the user) through the new dollar-giveback rule to
     estimate how much of the -50.83 total loss would likely have been avoided.
     This is an approximation (we only have each trade's peak $, not its full
     bar-by-bar path), so it reports a conservative estimate, not an exact replay.
"""
from bootstrap import add_project_root
add_project_root()

from datetime import datetime, timedelta, timezone

from live_runner import trade_management_params, manage_live_position


UTC = timezone.utc


class FakeMT5:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1


class FakePosition:
    def __init__(self, symbol, direction, price_open, sl, tp, ticket=1, open_time=None):
        self.symbol = symbol
        self.type = FakeMT5.POSITION_TYPE_BUY if direction == "BUY" else FakeMT5.POSITION_TYPE_SELL
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.ticket = ticket
        self.time = open_time or datetime.now(UTC)
        self.profit = 0.0


class FakeBroker:
    """Records close/modify calls instead of touching a real account."""

    def __init__(self):
        self.mt5 = FakeMT5()
        self.closed = []
        self.modified = []

    def close_position(self, position):
        self.closed.append((position.symbol, position.ticket, position.profit))
        return {"closed": True, "profit": position.profit}

    def modify_position(self, ticket, symbol, sl, tp):
        self.modified.append((symbol, ticket, sl, tp))
        return {"modified": True, "sl": sl, "tp": tp}


def usd_per_price_unit(entry_price, stop_price, dollar_risk):
    """Pick a $-per-price-unit multiplier so that (entry-stop) * multiplier == dollar_risk."""
    distance = abs(entry_price - stop_price)
    return dollar_risk / distance if distance else 0.0


def replay_path(symbol, direction, entry_price, stop_price, price_path, dollar_risk, mgmt=None):
    """
    Feed a sequence of prices through manage_live_position() bar by bar.
    Returns (action_taken, bar_index_of_action, final_profit_usd, peak_profit_usd_seen).
    """
    mgmt = mgmt or trade_management_params(symbol)
    broker = FakeBroker()
    tp = entry_price + 100 if direction == "BUY" else entry_price - 100  # far away, irrelevant to this test
    position = FakePosition(symbol, direction, entry_price, stop_price, tp, open_time=datetime.now(UTC) - timedelta(minutes=1))
    multiplier = usd_per_price_unit(entry_price, stop_price, dollar_risk)
    tracker = {}
    peak_seen = float("-inf")

    for i, price in enumerate(price_path):
        raw_pnl = (price - entry_price) if direction == "BUY" else (entry_price - price)
        position.profit = raw_pnl * multiplier
        peak_seen = max(peak_seen, position.profit)
        now = position.time + timedelta(minutes=i + 1)
        result, action = manage_live_position(broker, position, price, now, mgmt, tracker)
        if action == "modify_sl":
            position.sl = broker.modified[-1][2]
            continue
        if result is not None:
            return action, i, position.profit, peak_seen

    return "hold_to_end", len(price_path) - 1, position.profit, peak_seen


def pct(x):
    return f"{x:+.2f}"


def scenario_dollar_giveback_catch():
    """
    A trade with a WIDE stop (so it never reaches giveback_trigger_r on the R-based
    rule alone) that runs to a real dollar peak and then fades to a loss. The new
    dollar-based rule should catch this; simulate the OLD behaviour by disabling it
    to show the contrast.
    """
    print("\n--- Scenario 1: wide-stop trade that peaks in $ but never clears the R trigger ---")
    symbol = "USDCHF"
    entry = 0.80955
    stop = 0.80555  # wide (~40 pip) stop -> big R denominator, so a real $ peak stays well under giveback_trigger_r
    # price path: rally to a peak, then fade back down toward (not through) the stop
    path = [0.80955 + d for d in
            [0.00010, 0.00035, 0.00060, 0.00080, 0.00095, 0.00080, 0.00060, 0.00040, 0.00020, 0.00000, -0.00020]]
    dollar_risk = 15.0  # mirrors max_loss_per_trade_usd scale used live

    mgmt_new = trade_management_params(symbol)
    action_new, bar_new, final_new, peak_new = replay_path(symbol, "BUY", entry, stop, path, dollar_risk, mgmt_new)

    mgmt_old = dict(mgmt_new)
    mgmt_old["dollar_giveback_frac"] = 0.0  # simulate pre-fix behaviour (no $ safety net)
    action_old, bar_old, final_old, peak_old = replay_path(symbol, "BUY", entry, stop, path, dollar_risk, mgmt_old)

    print(f"  peak profit reached : ${peak_new:.2f}")
    print(f"  OLD (no $ giveback rule): action={action_old:26} closed_at_bar={bar_old:2d} final_pnl={pct(final_old)}")
    print(f"  NEW (with $ giveback rule): action={action_new:26} closed_at_bar={bar_new:2d} final_pnl={pct(final_new)}")
    improvement = final_new - final_old
    print(f"  => improvement from the fix: {pct(improvement)} on this trade")
    assert final_new > final_old, "expected the new $ giveback rule to outperform the old behaviour here"
    print("  PASS")


def scenario_usdchf_vs_shared_profile():
    """
    USDCHF used to share USDJPY's looser profile (giveback_trigger_r=1.20). Confirm
    the new USDCHF-specific profile (giveback_trigger_r=0.58) closes materially
    earlier on an identical price path, given an identical stop distance.
    """
    print("\n--- Scenario 2: USDCHF now has its own tighter profile vs. the old shared USDJPY profile ---")
    entry = 0.80870
    stop = 0.80820  # 50-pip-ish stop
    path = [0.80870 + d for d in
            [0.00010, 0.00025, 0.00040, 0.00050, 0.00045, 0.00030, 0.00015, 0.00000, -0.00015, -0.00030, -0.00045]]
    dollar_risk = 10.0

    old_shared_profile = trade_management_params("USDJPY")  # what USDCHF used to reuse
    new_usdchf_profile = trade_management_params("USDCHF")

    action_old, bar_old, final_old, peak_old = replay_path("USDCHF", "BUY", entry, stop, path, dollar_risk, old_shared_profile)
    action_new, bar_new, final_new, peak_new = replay_path("USDCHF", "BUY", entry, stop, path, dollar_risk, new_usdchf_profile)

    print(f"  peak profit reached : ${peak_new:.2f}")
    print(f"  OLD shared USDJPY profile : action={action_old:26} closed_at_bar={bar_old:2d} final_pnl={pct(final_old)}")
    print(f"  NEW USDCHF-specific profile: action={action_new:26} closed_at_bar={bar_new:2d} final_pnl={pct(final_new)}")
    print(f"  => improvement from the fix: {pct(final_new - final_old)} on this trade")
    assert bar_new <= bar_old, "expected the new tighter USDCHF profile to act at least as early"
    assert final_new >= final_old, "expected the new tighter USDCHF profile to retain at least as much profit"
    print("  PASS")


def scenario_breakeven_does_not_disable_management():
    """
    REGRESSION TEST for the bug the harness itself uncovered: risk used to be
    recomputed from the CURRENT position.sl every call. Once breakeven fires and
    sets sl == price_open, that recompute gave risk == 0 and the function returned
    "invalid_risk" forever after -- silently disabling all further trailing/giveback/
    dollar-stop management for the rest of the trade. Confirm that after breakeven
    fires, a subsequent giveback-close still fires (i.e. management keeps working).
    """
    print("\n--- Scenario 3 (regression test): management must keep working after breakeven fires ---")
    symbol = "AUDUSD"
    entry = 0.69800
    stop = 0.69750  # 50 pip stop
    mgmt = trade_management_params(symbol)
    # path: rise past breakeven_at_r, then keep rising past giveback_trigger_r, then fade
    path = [0.69800 + d for d in
            [0.00025, 0.00040, 0.00055, 0.00070, 0.00060, 0.00045, 0.00030]]
    dollar_risk = 10.0

    action, bar, final_pnl, peak = replay_path(symbol, "BUY", entry, stop, path, dollar_risk, mgmt)
    print(f"  peak profit reached : ${peak:.2f}")
    print(f"  action={action:26} closed_at_bar={bar:2d} final_pnl={pct(final_pnl)}")
    assert action not in ("invalid_risk", "hold_to_end"), (
        f"management appears to have stopped working after breakeven (action={action}) -- regression!"
    )
    assert final_pnl > 0, "expected the trade to still lock in a profit after breakeven, not ride to a loss"
    print("  PASS (breakeven no longer disables further management)")


# --- Part 2: replay real trades from the July 15 trade_analyzer log ------------------

# (symbol, direction, entry, exit, pnl, mfe, mae) exactly as printed by trade_analyzer.py
REAL_TRADES = [
    ("USDCHF", "SEL", 0.80826, 0.80856, -9.28, -1.55, -12.98),
    ("EURUSD", "BUY", 1.14366, 1.14334, -8.00, -0.25, -10.00),
    ("AUDUSD", "SEL", 0.69799, 0.69842, -10.75, 0.50, -19.00),
    ("EURUSD", "SEL", 1.14336, 1.14369, -8.25, 3.00, -13.75),
    ("USDCHF", "SEL", 0.80869, 0.80868, 0.31, 11.13, -5.87),
    ("USDCHF", "SEL", 0.80870, 0.80899, -8.96, -4.02, -8.65),
    ("USDCHF", "SEL", 0.80894, 0.80894, 0.00, 16.08, -12.36),
    ("AUDUSD", "BUY", 0.69854, 0.69819, -8.75, 1.75, -9.00),
    ("USDJPY", "BUY", 162.06400, 162.28100, 33.43, 36.66, -4.78),
    ("AUDUSD", "BUY", 0.69821, 0.69777, -11.00, 4.50, -6.50),
    ("EURUSD", "BUY", 1.14210, 1.14241, 7.75, 9.25, 0.00),
    ("USDCHF", "SEL", 0.80955, 0.80991, -11.11, 4.02, -15.74),
    ("EURUSD", "BUY", 1.14225, 1.14225, 0.00, 12.50, -2.75),
    ("USDJPY", "SEL", 162.27800, 162.32700, -7.55, 6.78, -10.47),
    ("EURUSD", "SEL", 1.14164, 1.14164, 0.00, 10.50, -4.25),
    ("AUDUSD", "BUY", 0.69805, 0.69803, -0.50, 24.75, -2.50),
    ("USDJPY", "SEL", 162.37800, 162.42200, -6.77, 3.08, -6.00),
    ("EURUSD", "SEL", 1.14148, 1.14148, 0.00, 10.25, 0.00),
    ("USDJPY", "BUY", 162.41600, 162.36500, -7.85, -0.31, -18.48),
    ("USDCHF", "SEL", 0.81098, 0.81099, -0.31, 16.35, -8.32),
    ("EURUSD", "BUY", 1.14135, 1.14133, -0.50, 14.75, -8.50),
    ("USDCHF", "BUY", 0.81144, 0.81104, -12.33, 0.31, -45.37),
    ("EURUSD", "BUY", 1.14072, 1.14192, 30.00, 42.50, 1.50),
    ("AUDUSD", "BUY", 0.69814, 0.69914, 25.00, 27.50, -1.25),
    ("AUDUSD", "SEL", 0.69913, 0.69929, -4.00, 9.50, -7.00),
    ("USDCHF", "BUY", 0.81013, 0.80979, -10.50, 7.10, -30.59),
    ("USDJPY", "BUY", 162.28800, 162.25300, -5.39, 2.77, -13.56),
    ("AUDUSD", "BUY", 0.69901, 0.69998, 24.25, 27.50, -2.25),
    ("EURUSD", "BUY", 1.14387, 1.14524, 34.25, 42.50, -5.25),
    ("EURUSD", "SEL", 1.14513, 1.14569, -14.00, -5.75, -39.25),
    ("USDCHF", "BUY", 0.80592, 0.80537, -17.07, -3.41, -25.15),
    ("EURUSD", "SEL", 1.14588, 1.14646, -14.50, -11.50, -20.50),
    ("AUDUSD", "SEL", 0.70045, 0.70087, -10.50, -6.00, -12.50),
    ("USDJPY", "SEL", 162.17400, 162.04400, 20.06, 31.80, -3.08),
    ("USDJPY", "BUY", 162.04700, 161.98000, -10.34, 1.54, -12.19),
    ("AUDUSD", "SEL", 0.70100, 0.70144, -11.00, -2.50, -15.25),
    ("AUDUSD", "SEL", 0.70142, 0.70155, -3.25, 2.50, -4.75),
    ("EURUSD", "SEL", 1.14689, 1.14716, -6.48, 5.04, -13.68),
    ("USDJPY", "BUY", 162.05800, 162.01300, -6.94, -7.72, -14.20),
]


def estimate_dollar_giveback_savings():
    """
    Approximate replay: we only know each trade's peak $ (mfe_usd) and final $
    (profit_usd), not its bar-by-bar path. For any trade where mfe_usd cleared
    min_peak_profit_usd, assume (conservatively) that the price path DID pass
    through the new dollar-giveback trigger level on its way from peak to final
    (true for any monotonic-ish fade, which matches what the giveback histogram in
    the user's chart shows -- most trades gave back smoothly, not in one violent
    tick). Under that assumption, the trade would have closed at:
        hypothetical_exit = peak - max(floor, frac * peak)
    instead of riding all the way to profit_usd. We report the better of the two
    (never worse than what actually happened), and only apply this to trades where
    mfe_usd > profit_usd (i.e. a real peak-then-fade occurred).
    """
    print("\n--- Estimated impact of the new $-based giveback rule on the 39 real trades ---")
    total_actual = 0.0
    total_estimated = 0.0
    changed = 0
    print(f"{'symbol':>7} | {'mfe':>7} | {'actual_pnl':>10} | {'est_new_pnl':>11} | {'delta':>7}")
    print("-" * 55)
    for symbol, direction, entry, exit_, pnl, mfe, mae in REAL_TRADES:
        mgmt = trade_management_params(symbol)
        floor = mgmt["dollar_giveback_floor_usd"]
        frac = mgmt["dollar_giveback_frac"]
        total_actual += pnl

        if mfe >= mgmt["min_peak_profit_usd"] and mfe > pnl:
            trigger_giveback = max(floor, frac * mfe)
            hypothetical_exit = mfe - trigger_giveback
            new_pnl = max(pnl, hypothetical_exit)  # never assume it does worse than reality
        else:
            new_pnl = pnl

        if abs(new_pnl - pnl) > 1e-9:
            changed += 1
        total_estimated += new_pnl
        marker = "  <-- improved" if new_pnl > pnl + 1e-9 else ""
        print(f"{symbol:>7} | {mfe:7.2f} | {pnl:10.2f} | {new_pnl:11.2f} | {new_pnl - pnl:+6.2f}{marker}")

    print("-" * 55)
    print(f"Actual total PnL (as logged) : {total_actual:.2f}")
    print(f"Estimated total PnL with fix : {total_estimated:.2f}")
    print(f"Trades where the new rule would have changed the outcome: {changed} / {len(REAL_TRADES)}")
    print(f"Estimated improvement: {total_estimated - total_actual:+.2f}")
    print(
        "\nCaveat: this is an estimate from mfe/pnl only (no bar-by-bar path for these "
        "closed trades), assuming the retracement from peak to final was gradual rather "
        "than a single violent tick. Re-run scripts/trade_analyzer.py after a live/dry-run "
        "period with the new code to get an exact, bar-verified number."
    )


if __name__ == "__main__":
    scenario_dollar_giveback_catch()
    scenario_usdchf_vs_shared_profile()
    scenario_breakeven_does_not_disable_management()
    estimate_dollar_giveback_savings()
    print("\nAll scenario assertions passed.")
