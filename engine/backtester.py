import pandas as pd


class Backtester:
    def __init__(self, data: pd.DataFrame, strategy, initial_balance=10000, spread=0.0002):
        self.data = data.copy()
        self.strategy = strategy

        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.spread = spread

        self.position = 0
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.risk = None
        self.bars_open = 0
        self.max_favorable_price = None

        self.trades = []
        self.equity_curve = []

    def run(self):
        signals = self.strategy.generate_signals(self.data)

        for i in range(len(self.data)):
            price = self.data["close"].iloc[i]
            atr = self.data["atr"].iloc[i]
            signal = signals.iloc[i]

            # ----------------------------
            # OPEN TRADE
            # ----------------------------
            if self.position == 0 and signal != 0 and pd.notna(atr):

                self.position = signal
                self.entry_price = price

                self.risk = 1.2 * atr  # SL distance
                self.bars_open = 0
                self.max_favorable_price = price

                if signal == 1:
                    self.sl = price - self.risk
                    self.tp = price + (2 * self.risk)
                else:
                    self.sl = price + self.risk
                    self.tp = price - (2 * self.risk)

            # ----------------------------
            # MANAGE TRADE
            # ----------------------------
            elif self.position != 0:
                self.bars_open += 1
                open_r = (price - self.entry_price) / self.risk if self.position == 1 else (self.entry_price - price) / self.risk

                # break-even after 0.3R
                if open_r >= 0.3:
                    if self.position == 1:
                        self.sl = max(self.sl, self.entry_price)
                    else:
                        self.sl = min(self.sl, self.entry_price)

                # trail after 0.6R
                if open_r >= 0.6:
                    trail_buffer = 0.3 * self.risk
                    if self.position == 1:
                        self.sl = max(self.sl, price - trail_buffer)
                    else:
                        self.sl = min(self.sl, price + trail_buffer)

                if self.bars_open >= 96:
                    self._close_trade(price, i)
                    self.equity_curve.append(self.balance)
                    continue

                # LONG
                if self.position == 1:
                    if price <= self.sl or price >= self.tp:
                        self._close_trade(price, i)

                # SHORT
                else:
                    if price >= self.sl or price <= self.tp:
                        self._close_trade(price, i)

            self.equity_curve.append(self.balance)

        return self.results()

    def _close_trade(self, price, i):
        if self.position == 1:
            pnl = (price - self.entry_price) / self.risk
        else:
            pnl = (self.entry_price - price) / self.risk

        pnl = pnl - self.spread

        self.balance += pnl * 100  # scaled R units

        self.trades.append({
            "type": "long" if self.position == 1 else "short",
            "entry": self.entry_price,
            "exit": price,
            "R": pnl,
            "bars": i
        })

        self.position = 0
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.risk = None
        self.bars_open = 0
        self.max_favorable_price = None

    def results(self):
        wins = len([t for t in self.trades if t["R"] > 0])

        avg_r = sum(t["R"] for t in self.trades) / len(self.trades) if self.trades else 0

        return {
            "final_balance": self.balance,
            "total_trades": len(self.trades),
            "win_rate": wins / len(self.trades) if self.trades else 0,
            "avg_r": avg_r,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }
