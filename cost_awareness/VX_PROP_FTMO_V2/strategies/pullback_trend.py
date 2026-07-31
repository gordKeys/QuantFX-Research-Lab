import pandas as pd
from strategies.base_strategy import BaseStrategy


class PullbackTrend(BaseStrategy):

    def __init__(self, trend_lookback=200, pullback_lookback=20, rsi_period=14, rsi_buy=45, rsi_sell=55):
        self.trend_lookback = trend_lookback
        self.pullback_lookback = pullback_lookback
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame):
        df = data.copy()
        signals = pd.Series(0, index=df.index)

        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        ema200 = df["close"].ewm(span=200, adjust=False).mean()
        rsi = self._rsi(df["close"])
        recent_low = df["low"].rolling(self.pullback_lookback).min()
        recent_high = df["high"].rolling(self.pullback_lookback).max()
        atr_ratio = df["atr"] / df["close"]

        for i in range(max(self.trend_lookback, self.pullback_lookback), len(df)):
            if pd.isna(rsi.iloc[i]) or pd.isna(atr_ratio.iloc[i]):
                continue

            if atr_ratio.iloc[i] < 0.0006:
                continue

            up_trend = ema50.iloc[i] > ema200.iloc[i] and ema50.iloc[i] > ema50.iloc[i - 3]
            down_trend = ema50.iloc[i] < ema200.iloc[i] and ema50.iloc[i] < ema50.iloc[i - 3]
            close_now = df["close"].iloc[i]

            if up_trend and close_now <= recent_low.iloc[i] * 1.002 and rsi.iloc[i] <= self.rsi_buy:
                signals.iloc[i] = 1
            elif down_trend and close_now >= recent_high.iloc[i] * 0.998 and rsi.iloc[i] >= self.rsi_sell:
                signals.iloc[i] = -1

        return signals
