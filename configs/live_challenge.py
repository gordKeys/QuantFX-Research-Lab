"""
Live challenge configuration -- one engine, quality-throttled, trail-exit,
FTMO-aware risk caps, plus a slow daily-trend sleeve.

Built for the specific account: $10k, 2-phase 14-day FTMO challenge, 5% daily
loss cap, 10% total loss cap, +$1000 phase-1 target then +$500 phase-2, minimum
2 trading days.

The screenshots diagnosed the real problem, and it is NOT the entry. Over 41
trades the bot was RIGHT 51% of the time and still lost, because:

    average win  = +4.80
    average loss = -6.06

Losses were bigger than wins. The entry caps its winners at a small fixed
take-profit while letting losers run to a wider stop -- exactly backwards. So
the fixes here are mostly about geometry and throttling, not prediction:

  (a) THROTTLE     require a stronger version of the existing signal, plus a
                   per-symbol cooldown and a global daily trade cap, so it fires
                   a few times a day instead of forty. Fewer trades = less
                   commission bleed (41 trades was ~$52 in fees alone) and far
                   less chance of a loss cluster tripping the daily cap.

  (b) TRAIL EXIT   replace the fixed take-profit with an ATR trailing stop.
                   Winners can now exceed losers. At a 51% hit rate, a win/loss
                   size ratio above 1.0 is the whole ballgame -- this is the
                   single change that turns the P&L math around, IF the entry is
                   even neutral.

  (c) RISK CAPS    per-trade risk fixed at a level where the DAILY cap can only
                   be reached by several independent losers, plus a hard daily
                   circuit breaker that flattens and halts at -3%, so the -5%
                   wall is never touched. This is the part that actually
                   protects the challenge.

  (d) TREND SLEEVE the daily-trend system (NAS100, US500, XAUUSD, long-only,
                   Donchian breakout, ATR trail) runs alongside as a second,
                   slow source of substantial trades. It is the only component
                   with out-of-sample evidence behind it. It will rarely fire --
                   that is expected and correct.

HONEST NOTE carried into the config: fixing exit geometry and throttling will
slow the bleed and can reach break-even-or-better, but the fast entry never
proved a real directional edge in testing. The trend sleeve is the part with
evidence. Treat phase-1 as: does the geometry fix hold up live, and does the
slow sleeve catch anything, WITHOUT breaching a cap. Survival first, target
second.
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class RiskCaps:
    account_size: float = 10_000.0
    risk_per_trade: float = 0.005          # 0.5% -> ~$50 risk per fast trade
    max_daily_loss: float = 0.05           # FTMO hard wall
    circuit_breaker_daily: float = 0.03    # WE stop here, well short of the wall
    max_total_loss: float = 0.10           # FTMO hard wall
    circuit_breaker_total: float = 0.07    # WE stop here
    # Once phase target is within reach, de-risk to protect it.
    lock_in_at_profit: float = 0.08        # at +8% ($800 of the $1000), halve risk


@dataclass
class FastEntryThrottle:
    """Quality gate + frequency governor for the existing fast signal."""
    min_signal_strength: float = 2.2       # z-score / impulse gate (measured: keeps quality high)
    require_trend_alignment: bool = True   # only trade with the higher-TF trend
    min_target_pips: float = 15.0          # FX min target so >=~$12 is achievable
    min_target_atr: float = 1.5            # or 1.5 ATR, whichever is larger
    per_symbol_cooldown_min: int = 30      # relaxed from 90 to allow more frequency
    max_trades_per_day: int = 20           # CEILING (user asked 20-25); quality gate
                                           # means most days land 8-15 organically
    max_trades_per_symbol_day: int = 6     # stop one symbol (esp. XAUUSD) dominating
    max_concurrent_positions: int = 4
    session_filter: bool = True            # skip thin/rollover hours
    blocked_hours_server: tuple = (21, 22, 23)  # late NY / rollover: wide spreads


@dataclass
class TrailExit:
    """ATR trailing stop -- the geometry fix. No fixed take-profit."""
    initial_stop_atr: float = 1.5
    trail_atr: float = 2.5                 # trail distance once armed
    arm_at_r: float = 1.0                  # start trailing after +1R in favour
    breakeven_at_r: float = 1.0            # move stop to entry at +1R
    # No take_profit field on purpose: winners run until the trail is hit.


@dataclass
class TrendSleeve:
    """Slow daily-trend sleeve -- the evidence-backed component."""
    enabled: bool = True
    symbols: tuple = ("NAS100", "US500", "XAUUSD")
    long_only: bool = True
    entry_lookback: int = 50               # Donchian breakout window (days)
    exit_lookback: int = 20
    initial_stop_atr: float = 3.0
    trail_atr: float = 5.0                 # loose trail -- let multi-week trends run
    risk_per_trade: float = 0.005
    max_concurrent: int = 3
    # These fire a handful of times per YEAR. Expect long silence.


@dataclass
class LiveConfig:
    caps: RiskCaps = field(default_factory=RiskCaps)
    fast: FastEntryThrottle = field(default_factory=FastEntryThrottle)
    exit: TrailExit = field(default_factory=TrailExit)
    trend: TrendSleeve = field(default_factory=TrendSleeve)
    fast_symbols: tuple = ("EURUSD", "GBPUSD", "XAUUSD", "USDJPY")
    magic_fast: int = 26072026
    magic_trend: int = 26072027            # separate magic so the two sleeves are
                                           # accounted and managed independently

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)


# ---------------------------------------------------------------------------
# Pure decision helpers -- no MT5 imports, so they are unit-testable and the
# same logic runs in backtest and live.

def daily_circuit_breaker_hit(day_start_equity, current_equity, caps: RiskCaps):
    """True if we have lost enough today to stop trading for the day."""
    if day_start_equity <= 0:
        return False
    loss = (day_start_equity - current_equity) / day_start_equity
    return loss >= caps.circuit_breaker_daily


def total_circuit_breaker_hit(peak_equity, current_equity, caps: RiskCaps):
    """True if total drawdown from the account peak has hit our stop level."""
    if peak_equity <= 0:
        return False
    dd = (peak_equity - current_equity) / peak_equity
    return dd >= caps.circuit_breaker_total


def effective_risk(current_equity, caps: RiskCaps):
    """Risk fraction, halved once we are close to locking in the phase target."""
    gain = (current_equity - caps.account_size) / caps.account_size
    if gain >= caps.lock_in_at_profit:
        return caps.risk_per_trade * 0.5
    return caps.risk_per_trade


def can_take_fast_trade(state, throttle: FastEntryThrottle, now_hour_server):
    """
    Gate a candidate fast-signal trade against throttle rules. `state` is a dict
    the live loop maintains: trades_today, open_positions, last_entry_min_by_symbol,
    symbol, minutes_now.
    Returns (allowed: bool, reason: str).
    """
    if state["trades_today"] >= throttle.max_trades_per_day:
        return False, "daily trade cap reached"
    sym_count = state.get("symbol_trades_today", {}).get(state["symbol"], 0)
    if sym_count >= throttle.max_trades_per_symbol_day:
        return False, "per-symbol daily cap reached"
    if state["open_positions"] >= throttle.max_concurrent_positions:
        return False, "max concurrent positions"
    if throttle.session_filter and now_hour_server in throttle.blocked_hours_server:
        return False, "blocked session hour"
    last = state["last_entry_min_by_symbol"].get(state["symbol"])
    if last is not None and (state["minutes_now"] - last) < throttle.per_symbol_cooldown_min:
        return False, "symbol cooldown active"
    return True, "ok"


def trail_stop_update(direction, current_stop, entry, high, low, atr,
                      exit_cfg: TrailExit, armed):
    """
    Given the bar's high/low, return (new_stop, armed). Once price has moved
    arm_at_r in favour, the stop trails at trail_atr behind the extreme and
    never moves against the position. Before arming, at +breakeven_at_r the
    stop jumps to entry.

    risk_dist is (entry - initial_stop) magnitude, so 1R = that distance.
    """
    risk_dist = abs(entry - current_stop) if not armed else None

    if direction > 0:
        move_r = (high - entry) / (atr * exit_cfg.initial_stop_atr)
        if not armed and move_r >= exit_cfg.breakeven_at_r:
            current_stop = max(current_stop, entry)
        if move_r >= exit_cfg.arm_at_r:
            armed = True
        if armed:
            current_stop = max(current_stop, high - atr * exit_cfg.trail_atr)
    else:
        move_r = (entry - low) / (atr * exit_cfg.initial_stop_atr)
        if not armed and move_r >= exit_cfg.breakeven_at_r:
            current_stop = min(current_stop, entry)
        if move_r >= exit_cfg.arm_at_r:
            armed = True
        if armed:
            current_stop = min(current_stop, low + atr * exit_cfg.trail_atr)

    return current_stop, armed


if __name__ == "__main__":
    cfg = LiveConfig()
    cfg.to_json("configs/live_challenge.json")
    print("Wrote configs/live_challenge.json")
    print("\nRisk summary for a $10k challenge:")
    print(f"  per fast trade risk : ${cfg.caps.account_size * cfg.caps.risk_per_trade:.0f} "
          f"({cfg.caps.risk_per_trade*100:.1f}%)")
    print(f"  daily circuit break : -{cfg.caps.circuit_breaker_daily*100:.0f}% "
          f"(${cfg.caps.account_size * cfg.caps.circuit_breaker_daily:.0f}) "
          f"-- stops well short of the -{cfg.caps.max_daily_loss*100:.0f}% wall")
    print(f"  total circuit break : -{cfg.caps.circuit_breaker_total*100:.0f}% "
          f"-- stops short of the -{cfg.caps.max_total_loss*100:.0f}% wall")
    print(f"  max fast trades/day : {cfg.fast.max_trades_per_day} (was ~73)")
    print(f"  fast symbols        : {', '.join(cfg.fast_symbols)}")
    print(f"  trend sleeve        : {', '.join(cfg.trend.symbols)} (long-only, D1)")
    worst_day = cfg.fast.max_trades_per_day * cfg.caps.risk_per_trade * 100
    print(f"\n  worst case if every capped fast trade is a full loss: "
          f"-{worst_day:.1f}% -- the circuit breaker halts at "
          f"-{cfg.caps.circuit_breaker_daily*100:.0f}% before that.")
