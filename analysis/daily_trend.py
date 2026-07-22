"""
Daily time-series momentum -- the one lead that cleared the probe AND matches a
real, documented, structurally-defensible effect.

Everything else in this project was an intraday pattern competing in the most
heavily mined arena in finance, and every one lost. This is different in kind:

  - It is the classic time-series-momentum / managed-futures effect (Moskowitz,
    Ooi & Pedersen 2012, and the whole CTA industry). Not a candle shape.
  - It survives specifically in equity indices and crypto because you cannot
    cheaply arbitrage away a broad multi-month uptrend, and because trend
    followers provide a service (absorbing risk in transitions) they get paid
    for. There is a mechanism, not just a backtest.
  - The regime probe flagged exactly the right instruments: NAS100 and US500
    cross 0.55 trend-persistence on D1 but NOT on H4, and crypto sits just
    under. A real slow effect shows up on the slow timeframe and vanishes on
    the fast one. That is the signature we want, and it is what we saw.

The entry is deliberately trivial, because in trend following the entry barely
matters -- the exit and the instrument selection do the work:

  ENTRY   price closes above the N-day high (Donchian breakout) for longs, below
          the N-day low for shorts. Optionally long-only, since equity indices
          have an upward structural drift that makes shorts a worse bet.

  STOP    a multiple of ATR below entry (there is no "structural" swing on a
          breakout the way there was on an impulse candle).

  EXIT    ATR trailing stop OR a close back through the opposite Donchian band,
          whichever comes first. This is what lets a multi-month trend run --
          the entire point of "few substantial trades".

Held on D1, signals are RARE -- a handful per instrument per year -- which is
the low-frequency, large-target profile asked for from the very first message.
"""

import numpy as np
import pandas as pd


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class DailyTrend:
    """
    Donchian-breakout time-series momentum.

    Parameters:
      entry_lookback   N-day high/low that defines a breakout
      exit_lookback    opposite-band lookback for the structural exit (shorter
                       than entry, so the exit is quicker than the entry)
      trail_atr        ATR-multiple trailing stop
      atr_stop         initial stop distance in ATR
      long_only        skip shorts (sensible for equity indices)
    """

    def __init__(self, entry_lookback=50, exit_lookback=20, trail_atr=4.0,
                 atr_stop=3.0, long_only=False):
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
        self.trail_atr = trail_atr
        self.atr_stop = atr_stop
        self.long_only = long_only

    def as_dict(self):
        return {
            "entry_lookback": self.entry_lookback,
            "exit_lookback": self.exit_lookback,
            "trail_atr": self.trail_atr,
            "atr_stop": self.atr_stop,
            "long_only": self.long_only,
        }

    def entries(self, df, atr_values):
        """
        Emit (bar_index, direction, initial_stop). A breakout fires when the
        close exceeds the prior N-day high (long) or low (short). The prior-day
        extreme is used so the signal is causal -- no same-bar lookahead.
        """
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(df)

        # prior N-day high/low, shifted so bar i sees only bars < i
        roll_high = pd.Series(high).rolling(self.entry_lookback).max().shift(1).to_numpy()
        roll_low = pd.Series(low).rolling(self.entry_lookback).min().shift(1).to_numpy()

        entries = []
        in_trade_until = -1

        for i in range(self.entry_lookback + 1, n):
            band = atr_values[i]
            if np.isnan(band) or band <= 0 or i <= in_trade_until:
                continue
            if not np.isnan(roll_high[i]) and close[i] > roll_high[i]:
                entries.append((i, 1, close[i] - band * self.atr_stop))
                in_trade_until = i  # the backtester enforces one-at-a-time
            elif not self.long_only and not np.isnan(roll_low[i]) and close[i] < roll_low[i]:
                entries.append((i, -1, close[i] + band * self.atr_stop))
                in_trade_until = i
        return entries
