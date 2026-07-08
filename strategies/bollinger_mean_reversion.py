import pandas as pd
from strategies.base_strategy import BaseStrategy


class BollingerMeanReversion(BaseStrategy):

    def __init__(self, lookback=20, std_dev=2.0, min_atr_ratio=0.00045):
        self.lookback = lookback
        self.std_dev = std_dev
        self.min_atr_ratio = min_atr_ratio

    def generate_signals(self, data: pd.DataFrame):
        df = data.copy()
        signals = pd.Series(0, index=df.index)

        middle = df["close"].rolling(self.lookback).mean()
        stdev = df["close"].rolling(self.lookback).std()
        upper = middle + self.std_dev * stdev
        lower = middle - self.std_dev * stdev
        atr_ratio = df["atr"] / df["close"]
        ema50 = df["ema50"]
        ema200 = df["ema200"]

        for i in range(self.lookback, len(df)):
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]) or pd.isna(atr_ratio.iloc[i]):
                continue

            if atr_ratio.iloc[i] < self.min_atr_ratio:
                continue

            close_now = df["close"].iloc[i]
            trend_bias = abs((ema50.iloc[i] - ema200.iloc[i]) / close_now)
            if trend_bias > 0.0022:
                continue

            if close_now > upper.iloc[i]:
                signals.iloc[i] = -1
            elif close_now < lower.iloc[i]:
                signals.iloc[i] = 1

        return signals
