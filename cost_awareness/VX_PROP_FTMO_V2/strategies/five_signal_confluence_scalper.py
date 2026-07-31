import pandas as pd

from strategies.base_strategy import BaseStrategy


class FiveSignalConfluenceScalper(BaseStrategy):

    COMPONENTS = ("trend", "band_extreme", "rsi_extreme", "candle_pattern", "volume_spike", "support_resistance")

    def __init__(
        self,
        lookback=20,
        support_lookback=30,
        volume_lookback=20,
        rsi_period=14,
        min_score=3,
        require_trend_alignment=False,
        disabled_components=None,
    ):
        self.lookback = lookback
        self.support_lookback = support_lookback
        self.volume_lookback = volume_lookback
        self.rsi_period = rsi_period
        self.min_score = min_score
        # Structural hypotheses to test, not just the score threshold:
        # - require_trend_alignment: only take a signal if the EMA trend
        #   component agrees with the direction that scored highest, instead
        #   of letting trend and mean-reversion components (RSI/band/S-R,
        #   which fire on oversold/overbought extremes that often coincide
        #   with a counter-trend move) get sumed into one undifferentiated
        #   score where they can silently cancel out or reinforce a
        #   direction the trend filter itself disagrees with.
        # - disabled_components: component names (from COMPONENTS) to exclude
        #   from scoring entirely, to test whether a component is dead
        #   weight or actively harmful for a given symbol.
        self.require_trend_alignment = require_trend_alignment
        self.disabled_components = set(disabled_components or [])
        # Populated by generate_signals with the long/short confluence score
        # of the most recently processed bar, so callers (live_runner) can
        # log the entry score alongside the trade without recomputing it.
        self.last_long_score = None
        self.last_short_score = None
        # Populated by generate_signals with a per-bar record of which
        # components fired in the winning direction, keyed by bar timestamp,
        # for post-hoc analysis of which components actually predict
        # winners (see scripts/entry_quality_analyzer.py). Only bars with a
        # non-zero signal are recorded.
        self.last_run_components = {}

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame):
        df = data.copy()
        signals = pd.Series(0, index=df.index)
        self.last_run_components = {}

        ema_fast = df["close"].ewm(span=20, adjust=False).mean()
        ema_slow = df["close"].ewm(span=50, adjust=False).mean()
        middle = df["close"].rolling(self.lookback).mean()
        stdev = df["close"].rolling(self.lookback).std()
        upper = middle + 2 * stdev
        lower = middle - 2 * stdev
        rsi = self._rsi(df["close"])
        avg_volume = df["tick_volume"].rolling(self.volume_lookback).mean()
        support = df["low"].rolling(self.support_lookback).min().shift(1)
        resistance = df["high"].rolling(self.support_lookback).max().shift(1)

        for i in range(max(self.lookback, self.support_lookback, self.volume_lookback, self.rsi_period), len(df)):
            if any(pd.isna(val) for val in [ema_fast.iloc[i], ema_slow.iloc[i], upper.iloc[i], lower.iloc[i], rsi.iloc[i], avg_volume.iloc[i], support.iloc[i], resistance.iloc[i]]):
                continue

            close_now = df["close"].iloc[i]
            high_now = df["high"].iloc[i]
            low_now = df["low"].iloc[i]
            open_now = df["open"].iloc[i]
            volume_now = df["tick_volume"].iloc[i]
            body = abs(close_now - open_now)
            candle_range = max(high_now - low_now, 1e-9)
            upper_wick = high_now - max(open_now, close_now)
            lower_wick = min(open_now, close_now) - low_now
            volume_spike = volume_now > avg_volume.iloc[i] * 1.2

            bullish_engulfing = (
                close_now > open_now
                and df["close"].iloc[i - 1] < df["open"].iloc[i - 1]
                and close_now >= df["open"].iloc[i - 1]
                and open_now <= df["close"].iloc[i - 1]
            )
            bearish_engulfing = (
                close_now < open_now
                and df["close"].iloc[i - 1] > df["open"].iloc[i - 1]
                and open_now >= df["close"].iloc[i - 1]
                and close_now <= df["open"].iloc[i - 1]
            )
            pin_bar_bull = lower_wick > body * 2 and lower_wick / candle_range > 0.45 and close_now > open_now
            pin_bar_bear = upper_wick > body * 2 and upper_wick / candle_range > 0.45 and close_now < open_now

            long_flags = {
                "trend": ema_fast.iloc[i] > ema_slow.iloc[i],
                "band_extreme": close_now < lower.iloc[i],
                "rsi_extreme": rsi.iloc[i] <= 35,
                "candle_pattern": bullish_engulfing or pin_bar_bull,
                "volume_spike": volume_spike and close_now > open_now,
                "support_resistance": close_now <= support.iloc[i] * 1.0015,
            }
            short_flags = {
                "trend": ema_fast.iloc[i] < ema_slow.iloc[i],
                "band_extreme": close_now > upper.iloc[i],
                "rsi_extreme": rsi.iloc[i] >= 65,
                "candle_pattern": bearish_engulfing or pin_bar_bear,
                "volume_spike": volume_spike and close_now < open_now,
                "support_resistance": close_now >= resistance.iloc[i] * 0.9985,
            }

            long_score = sum(1 for name, active in long_flags.items() if active and name not in self.disabled_components)
            short_score = sum(1 for name, active in short_flags.items() if active and name not in self.disabled_components)

            direction = 0
            if long_score >= self.min_score and long_score > short_score:
                if not self.require_trend_alignment or long_flags["trend"]:
                    direction = 1
            elif short_score >= self.min_score and short_score > long_score:
                if not self.require_trend_alignment or short_flags["trend"]:
                    direction = -1

            if direction != 0:
                signals.iloc[i] = direction
                active_flags = long_flags if direction == 1 else short_flags
                self.last_run_components[df.index[i]] = {
                    name: bool(active) for name, active in active_flags.items()
                }

            if i == len(df) - 1:
                self.last_long_score = long_score
                self.last_short_score = short_score

        return signals
