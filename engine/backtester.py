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
        self.max_favorable_pnl = 0.0
        self.risk_per_trade = 0.0025
        self.breakeven_at_r = 1.25
        self.trail_at_r = 2.25
        self.trail_buffer_r = 0.75
        self.max_bars_loss_cut = 16
        self.profit_floor_r = 2.25
        self.profit_fade_pct = 0.20

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
                self.max_favorable_pnl = 0.0

                if signal == 1:
                    self.sl = price - self.risk
                    self.tp = price + (3 * self.risk)
                else:
                    self.sl = price + self.risk
                    self.tp = price - (3 * self.risk)

            # ----------------------------
            # MANAGE TRADE
            # ----------------------------
            elif self.position != 0:
                self.bars_open += 1
                current_pnl = (price - self.entry_price) if self.position == 1 else (self.entry_price - price)
                self.max_favorable_pnl = max(self.max_favorable_pnl, current_pnl)
                open_r = current_pnl / self.risk

                if self.max_favorable_pnl >= self.profit_floor_r * self.risk and current_pnl <= self.max_favorable_pnl * (1 - self.profit_fade_pct):
                    self._close_trade(price, i, status="PROFIT_FADE")
                    self.equity_curve.append(self.balance)
                    continue

                if self.bars_open >= self.max_bars_loss_cut and open_r <= -0.30:
                    self._close_trade(price, i, status="LOSS_CUT")
                    self.equity_curve.append(self.balance)
                    continue

                if open_r >= self.breakeven_at_r:
                    if self.position == 1:
                        self.sl = max(self.sl, self.entry_price)
                    else:
                        self.sl = min(self.sl, self.entry_price)

                if open_r >= self.trail_at_r:
                    trail_buffer = self.trail_buffer_r * self.risk
                    if self.position == 1:
                        self.sl = max(self.sl, price - trail_buffer)
                    else:
                        self.sl = min(self.sl, price + trail_buffer)

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

    def _close_trade(self, price, i, status="EXIT"):
        if self.position == 1:
            pnl = (price - self.entry_price) / self.risk
        else:
            pnl = (self.entry_price - price) / self.risk

        pnl = pnl - self.spread

        self.balance += pnl * (self.initial_balance * self.risk_per_trade)

        self.trades.append({
            "type": "long" if self.position == 1 else "short",
            "entry": self.entry_price,
            "exit": price,
            "R": pnl,
            "bars": i,
            "status": status,
        })

        self.position = 0
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.risk = None
        self.bars_open = 0
        self.max_favorable_pnl = 0.0

    def results(self):
        wins = [t["R"] for t in self.trades if t["R"] > 0]
        losses = [t["R"] for t in self.trades if t["R"] < 0]
        total_win = sum(wins)
        total_loss = abs(sum(losses))
        avg_r = sum(t["R"] for t in self.trades) / len(self.trades) if self.trades else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = (total_win / total_loss) if total_loss else (float("inf") if total_win > 0 else 0.0)
        expectancy = avg_r

        return {
            "final_balance": self.balance,
            "total_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0,
            "avg_r": avg_r,
            "avg_win_r": avg_win,
            "avg_loss_r": avg_loss,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }
