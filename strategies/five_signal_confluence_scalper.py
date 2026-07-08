import pandas as pd

from strategies.base_strategy import BaseStrategy


class FiveSignalConfluenceScalper(BaseStrategy):

    def __init__(
        self,
        lookback=20,
        support_lookback=30,
        volume_lookback=20,
        rsi_period=14,
        min_score=3,
    ):
        self.lookback = lookback
        self.support_lookback = support_lookback
        self.volume_lookback = volume_lookback
        self.rsi_period = rsi_period
        self.min_score = min_score

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame):
        df = data.copy()
        signals = pd.Series(0, index=df.index)

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

            long_score = 0
            short_score = 0

            if ema_fast.iloc[i] > ema_slow.iloc[i]:
                long_score += 1
            elif ema_fast.iloc[i] < ema_slow.iloc[i]:
                short_score += 1

            if close_now < lower.iloc[i]:
                long_score += 1
            if close_now > upper.iloc[i]:
                short_score += 1

            if rsi.iloc[i] <= 35:
                long_score += 1
            if rsi.iloc[i] >= 65:
                short_score += 1

            if bullish_engulfing or pin_bar_bull:
                long_score += 1
            if bearish_engulfing or pin_bar_bear:
                short_score += 1

            if volume_spike:
                if close_now > open_now:
                    long_score += 1
                elif close_now < open_now:
                    short_score += 1

            near_support = close_now <= support.iloc[i] * 1.0015
            near_resistance = close_now >= resistance.iloc[i] * 0.9985
            if near_support:
                long_score += 1
            if near_resistance:
                short_score += 1

            if long_score >= self.min_score and long_score > short_score:
                signals.iloc[i] = 1
            elif short_score >= self.min_score and short_score > long_score:
                signals.iloc[i] = -1

        return signals
