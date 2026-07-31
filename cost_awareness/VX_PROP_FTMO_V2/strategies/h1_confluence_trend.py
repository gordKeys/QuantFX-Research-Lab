import pandas as pd
from strategies.base_strategy import BaseStrategy


class H1ConfluenceTrend(BaseStrategy):

    def __init__(
        self,
        h1_trend_span=20,
        rsi_period=14,
        atr_period=14,
        min_score=3,
    ):
        self.h1_trend_span = h1_trend_span
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.min_score = min_score

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    def _resample_h1(self, df: pd.DataFrame) -> pd.DataFrame:
        h1 = df.resample("1h").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
                "spread": "mean",
                "real_volume": "sum",
            }
        )
        return h1.dropna()

    def _build_h1_map(self, df: pd.DataFrame) -> pd.DataFrame:
        h1 = self._resample_h1(df)
        if h1.empty or len(h1) < 50:
            return pd.DataFrame(index=df.index)

        h1["ema20"] = h1["close"].ewm(span=self.h1_trend_span, adjust=False).mean()
        h1["ema50"] = h1["close"].ewm(span=50, adjust=False).mean()
        h1["ema200"] = h1["close"].ewm(span=200, adjust=False).mean()
        h1["adx_proxy"] = (h1["ema20"] - h1["ema50"]).abs() / h1["close"]
        h1["trend_up"] = (h1["ema20"] > h1["ema200"]) & (h1["adx_proxy"] > 0.001)
        h1["trend_down"] = (h1["ema20"] < h1["ema200"]) & (h1["adx_proxy"] > 0.001)

        h1_rsi = self._rsi(h1["close"])
        h1_atr = pd.concat(
            [
                h1["high"] - h1["low"],
                (h1["high"] - h1["close"].shift()).abs(),
                (h1["low"] - h1["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1).rolling(self.atr_period).mean()

        h1_bb_mid = h1["close"].rolling(20).mean()
        h1_bb_std = h1["close"].rolling(20).std()
        h1_bb_upper = h1_bb_mid + 2 * h1_bb_std
        h1_bb_lower = h1_bb_mid - 2 * h1_bb_std

        h1_map = pd.DataFrame(index=df.index)
        h1_map["trend_up"] = h1["trend_up"].reindex(df.index, method="ffill")
        h1_map["trend_down"] = h1["trend_down"].reindex(df.index, method="ffill")
        h1_map["rsi"] = h1_rsi.reindex(df.index, method="ffill")
        h1_map["atr"] = h1_atr.reindex(df.index, method="ffill")
        h1_map["bb_upper"] = h1_bb_upper.reindex(df.index, method="ffill")
        h1_map["bb_lower"] = h1_bb_lower.reindex(df.index, method="ffill")
        return h1_map

    def _session_allowed(self, ts: pd.Timestamp) -> bool:
        return True

    def generate_signals(self, data: pd.DataFrame):
        df = data.copy()
        signals = pd.Series(0, index=df.index)
        h1_map = self._build_h1_map(df)
        if h1_map.empty:
            return signals

        for i in range(max(50, self.atr_period, self.rsi_period), len(df)):
            if not self._session_allowed(df.index[i]):
                continue
            if pd.isna(h1_map["atr"].iloc[i]) or pd.isna(h1_map["rsi"].iloc[i]):
                continue

            trend_up = bool(h1_map["trend_up"].iloc[i])
            trend_down = bool(h1_map["trend_down"].iloc[i])
            rsi = float(h1_map["rsi"].iloc[i])
            close_now = float(df["close"].iloc[i])
            open_now = float(df["open"].iloc[i])
            high_now = float(df["high"].iloc[i])
            low_now = float(df["low"].iloc[i])
            prev_close = float(df["close"].iloc[i - 1])
            bb_upper = h1_map["bb_upper"].iloc[i]
            bb_lower = h1_map["bb_lower"].iloc[i]
            score_buy = 0
            score_sell = 0

            ema_fast = df["close"].ewm(span=20, adjust=False).mean().iloc[i]
            ema_slow = df["close"].ewm(span=50, adjust=False).mean().iloc[i]
            if ema_fast > ema_slow:
                score_buy += 1
            elif ema_fast < ema_slow:
                score_sell += 1

            macd_fast = df["close"].ewm(span=12, adjust=False).mean()
            macd_slow = df["close"].ewm(span=26, adjust=False).mean()
            macd_hist = (macd_fast - macd_slow) - (macd_fast - macd_slow).ewm(span=9, adjust=False).mean()
            if macd_hist.iloc[i] > 0:
                score_buy += 1
            elif macd_hist.iloc[i] < 0:
                score_sell += 1

            if not pd.isna(bb_lower) and close_now <= bb_lower and rsi <= 35:
                score_buy += 1
            if not pd.isna(bb_upper) and close_now >= bb_upper and rsi >= 65:
                score_sell += 1

            bullish_engulfing = close_now > open_now and prev_close < df["open"].iloc[i - 1] and close_now >= df["open"].iloc[i - 1]
            bearish_engulfing = close_now < open_now and prev_close > df["open"].iloc[i - 1] and close_now <= df["open"].iloc[i - 1]
            if bullish_engulfing or (close_now > open_now and (close_now - low_now) > (high_now - close_now) * 1.2):
                score_buy += 1
            if bearish_engulfing or (close_now < open_now and (high_now - close_now) > (close_now - low_now) * 1.2):
                score_sell += 1

            vol_ma = df["tick_volume"].rolling(20).mean().iloc[i]
            if not pd.isna(vol_ma) and df["tick_volume"].iloc[i] > vol_ma * 1.2:
                if trend_up:
                    score_buy += 1
                if trend_down:
                    score_sell += 1

            if trend_up and score_buy >= self.min_score:
                signals.iloc[i] = 1
            elif trend_down and score_sell >= self.min_score:
                signals.iloc[i] = -1

        return signals


class H1SessionConfluenceTrend(H1ConfluenceTrend):

    def __init__(self, allowed_hours=None, **kwargs):
        super().__init__(**kwargs)
        self.allowed_hours = allowed_hours or {11, 13, 14, 16, 18, 20}

    def _session_allowed(self, ts: pd.Timestamp) -> bool:
        return ts.hour in self.allowed_hours
