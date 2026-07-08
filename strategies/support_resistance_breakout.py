import pandas as pd
from strategies.base_strategy import BaseStrategy


class SupportResistanceBreakout(BaseStrategy):

    def __init__(self, lookback=30, breakout_buffer=0.0006, min_atr_ratio=0.00055):
        self.lookback = lookback
        self.breakout_buffer = breakout_buffer
        self.min_atr_ratio = min_atr_ratio

    def generate_signals(self, data: pd.DataFrame):
        df = data.copy()
        signals = pd.Series(0, index=df.index)

        resistance = df["high"].rolling(self.lookback).max().shift(1)
        support = df["low"].rolling(self.lookback).min().shift(1)
        atr_ratio = df["atr"] / df["close"]
        ema50 = df["ema50"]
        ema200 = df["ema200"]

        for i in range(self.lookback + 1, len(df)):
            if pd.isna(resistance.iloc[i]) or pd.isna(support.iloc[i]) or pd.isna(atr_ratio.iloc[i]):
                continue

            if atr_ratio.iloc[i] < self.min_atr_ratio:
                continue

            close_now = df["close"].iloc[i]
            prev_close = df["close"].iloc[i - 1]
            trend_bias = abs((ema50.iloc[i] - ema200.iloc[i]) / close_now)
            if trend_bias < 0.0012:
                continue

            if close_now > resistance.iloc[i] * (1 + self.breakout_buffer) and prev_close <= resistance.iloc[i]:
                signals.iloc[i] = 1
            elif close_now < support.iloc[i] * (1 - self.breakout_buffer) and prev_close >= support.iloc[i]:
                signals.iloc[i] = -1

        return signals
