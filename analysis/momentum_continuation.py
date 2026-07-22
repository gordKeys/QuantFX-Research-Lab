"""
Momentum-continuation strategy -- what the data actually pointed to.

The long road here: four SMC components tested and killed, then a "fair value
gap" effect that survived six controls, then decomposed into (a) mostly drift
and (b) a residual that turned out not to need gaps at all. Entering after any
fast >1 ATR impulse reproduced the effect with no gap structure. So this is the
honest version of the signal -- short-horizon momentum after a decisive move.

Three deliberate departures from the barrier tests that produced it:

  ENTRY     A confirmed impulse: one bar closing more than `impulse_atr` ATR
            beyond its open, in the trend direction (price above/below a slow
            EMA). Continuation, not reversal.

  STOP      Structural, not fixed. The low (for longs) of the impulse candle,
            padded slightly. That candle is the origin of the move; if price
            trades back through it the premise is void. A swept structural stop
            is a real invalidation, unlike an arbitrary 1.5 ATR line that sits
            wherever the arithmetic lands.

  EXIT      ATR trailing stop, NOT a fixed take-profit. The barrier test capped
            winners at +k ATR and gave the edge back -- marked to money,
            buy-and-hold beat it because the cap threw away the continuation.
            A trail lets the substantial trades run, which is the entire point
            of "few substantial trades" and "don't let winners turn into
            losers". This is the giveback protection asked for at the very
            start of the project, in its correct form.

This module is intentionally exit-heavy and entry-light. The research said the
entry is modest and the exit is where money is kept or lost, so that is where
the logic lives.
"""

import numpy as np
import pandas as pd


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def find_entries(df, atr_values, impulse_atr=1.0, trend_ema=50, cooldown=10):
    """
    Emit (bar_index, direction, stop_price) for each confirmed impulse.

    An entry needs three things, all known at the signal bar:
      1. the bar's body exceeds impulse_atr * ATR
      2. it closes on the trend side of a slow EMA (continuation filter)
      3. no entry fired in the last `cooldown` bars (avoid stacking one move)

    The stop is the opposite extreme of the impulse candle. Entry is assumed at
    the next bar's open by the backtester, never at this bar's close.
    """
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ema = df["close"].ewm(span=trend_ema, adjust=False).mean().to_numpy()

    entries = []
    last_fired = -10 ** 9

    for i in range(trend_ema, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0 or i - last_fired < cooldown:
            continue

        body = close[i] - open_[i]
        if abs(body) < band * impulse_atr:
            continue

        if body > 0 and close[i] > ema[i]:
            entries.append((i, 1, low[i]))
            last_fired = i
        elif body < 0 and close[i] < ema[i]:
            entries.append((i, -1, high[i]))
            last_fired = i

    return entries


class MomentumContinuation:
    """
    Parameter bundle plus the entry generator. The backtester owns the exit
    walk (trailing stop, structural stop, costs), because the exit needs bar-by
    -bar state the strategy object should not hold.
    """

    def __init__(self, impulse_atr=1.0, trend_ema=50, cooldown=10,
                 trail_atr=2.0, stop_pad_atr=0.1):
        self.impulse_atr = impulse_atr
        self.trend_ema = trend_ema
        self.cooldown = cooldown
        self.trail_atr = trail_atr          # trailing stop distance, in ATR
        self.stop_pad_atr = stop_pad_atr    # padding below the structural stop

    def entries(self, df, atr_values):
        return find_entries(df, atr_values, self.impulse_atr,
                            self.trend_ema, self.cooldown)

    def as_dict(self):
        return {
            "impulse_atr": self.impulse_atr,
            "trend_ema": self.trend_ema,
            "cooldown": self.cooldown,
            "trail_atr": self.trail_atr,
            "stop_pad_atr": self.stop_pad_atr,
        }
