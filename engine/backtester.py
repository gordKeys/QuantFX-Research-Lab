import pandas as pd

from engine.cost_model import (
    approx_lots,
    commission_cost_usd,
    load_commission_map,
    load_swap_map,
    nights_held,
    spread_cost_price,
    swap_cost_usd,
)


class Backtester:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        initial_balance=10000,
        spread=0.0002,
        symbol=None,
        commission_map=None,
        commission_per_lot=None,
        slippage_points=0.0,
        swap_map=None,
        swap_per_lot_per_night=None,
        rollover_hour_utc=21,
    ):
        """
        spread: fallback flat PRICE-space round-trip cost, used only when
            `symbol` is None, or the bar's real 'spread' column is missing /
            zero. If `symbol` is given and `data` has a 'spread' column
            (MT5 always exports one), the REAL per-bar spread is used
            instead -- this is what actually varies your live fills, not a
            single guessed number.

        symbol: e.g. "XAUUSD". Enables real per-bar spread (via
            engine.cost_model) and commission costing. Leave None to keep
            the old flat-spread-only behavior (still bug-fixed, see below).

        commission_map: optional {"EURUSD": 5.04, ...} of USD per lot round
            trip, e.g. from configs/commission_map.json
            (scripts/measure_commission.py). Auto-loaded from that path if
            left as None and the file exists.

        commission_per_lot: explicit override, takes priority over
            commission_map and the built-in asset-class fallback.

        slippage_points: extra round-trip cost in the same point units as
            the 'spread' column, added on top of spread to approximate fills
            that don't land exactly on your signal price. Defaults to 0
            (off) since it's a guess, not measured -- turn it on deliberately.

        swap_map: optional {"EURUSD": -3.2, ...} of USD per lot per night
            held (measured empirically, e.g. from
            configs/swap_map.json). Auto-loaded from that path if left as
            None and the file exists. No default fallback if missing --
            unmeasured swap is treated as 0, not guessed (see
            engine.cost_model.load_swap_map for why).

        swap_per_lot_per_night: explicit override, takes priority over
            swap_map.

        rollover_hour_utc: the hour (UTC) your broker charges swap at --
            confirm against your own account; 21:00-22:00 UTC is typical
            but broker/DST-dependent. Only matters for trades held across
            that boundary; same-session trades pay no swap regardless.

        NOTE ON THE BUG THIS FIXES: the previous version did
            `pnl = pnl - self.spread` where `pnl` is already in R-multiples
            (risk-normalized) but `self.spread` was a raw PRICE distance
            (e.g. 0.0002). Those are different units -- on XAUUSD in
            particular (price ~2000-4700, R measured in dollars-per-ATR),
            subtracting a fixed FX-sized number in R-space did essentially
            nothing close to the real cost. Spread/commission are now
            converted into the same R-units as the trade before being
            subtracted.
        """
        self.data = data.copy()
        self.strategy = strategy

        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.spread = spread  # flat PRICE-space fallback only

        self.symbol = symbol
        self.commission_map = (
            commission_map if commission_map is not None else load_commission_map()
        )
        self.commission_per_lot = commission_per_lot
        self.slippage_points = slippage_points
        self.swap_map = swap_map if swap_map is not None else load_swap_map()
        self.swap_per_lot_per_night = swap_per_lot_per_night
        self.rollover_hour_utc = rollover_hour_utc

        self.position = 0
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.risk = None
        self.bars_open = 0
        self.risk_per_trade = 0.0025
        self.breakeven_at_r = 2.00
        self.trail_at_r = 4.00
        self.trail_buffer_r = 1.25
        self.max_bars_loss_cut = 24

        self.trades = []
        self.equity_curve = []
        self._entry_spread_price = None
        self._entry_time = None

        # Cost accounting, tracked separately so you can see how much of
        # your gross edge costs actually ate, not just the net number.
        self.total_spread_cost_usd = 0.0
        self.total_commission_usd = 0.0
        self.total_swap_usd = 0.0

    def run(self):
        signals = self.strategy.generate_signals(self.data)
        has_spread_col = "spread" in self.data.columns

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
                self._entry_time = self.data.index[i]

                self.risk = 1.2 * atr  # SL distance
                self.bars_open = 0
                if signal == 1:
                    self.sl = price - self.risk
                    self.tp = price + (3 * self.risk)
                else:
                    self.sl = price + self.risk
                    self.tp = price - (3 * self.risk)

                # Spread is charged the instant a position opens (you're
                # down the full spread immediately) -- so it's priced off
                # the entry bar, not the exit bar.
                spread_points = self.data["spread"].iloc[i] if has_spread_col else None
                self._entry_spread_price = spread_cost_price(
                    self.symbol, spread_points, fallback_price=self.spread
                )
                if self.slippage_points:
                    from engine.cost_model import POINT_SIZE, DEFAULT_POINT_SIZE

                    point = POINT_SIZE.get((self.symbol or "").upper(), DEFAULT_POINT_SIZE)
                    self._entry_spread_price += self.slippage_points * point

            # ----------------------------
            # MANAGE TRADE
            # ----------------------------
            elif self.position != 0:
                self.bars_open += 1
                current_pnl = (price - self.entry_price) if self.position == 1 else (self.entry_price - price)
                open_r = current_pnl / self.risk

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
            gross_r = (price - self.entry_price) / self.risk
        else:
            gross_r = (self.entry_price - price) / self.risk

        risk_usd = self.initial_balance * self.risk_per_trade

        # --- Spread cost, converted into R-units (same space as gross_r) ---
        spread_price_cost = self._entry_spread_price if self._entry_spread_price is not None else self.spread
        spread_cost_r = spread_price_cost / self.risk if self.risk else 0.0
        spread_cost_usd_amt = spread_cost_r * risk_usd

        # --- Commission cost, priced in USD then converted into R-units ---
        lots = approx_lots(self.symbol, risk_usd, self.risk, self.entry_price)
        commission_usd_amt = commission_cost_usd(
            self.symbol, lots, self.commission_map, self.commission_per_lot
        )
        commission_cost_r = commission_usd_amt / risk_usd if risk_usd else 0.0

        # --- Swap cost, only nonzero for trades held across a rollover ---
        # NOTE ON SIGN: unlike spread/commission (always a cost, always
        # subtracted), swap_usd_amt is SIGNED -- negative when it's a real
        # cost (the common case), positive on the rare occasion a broker
        # credits swap for holding a particular direction. So it's ADDED
        # here, not subtracted; a negative swap_usd_amt still reduces net_r.
        exit_time = self.data.index[i]
        held_nights = nights_held(self._entry_time, exit_time, rollover_hour_utc=self.rollover_hour_utc)
        swap_usd_amt = swap_cost_usd(
            self.symbol, lots, held_nights, self.swap_map, self.swap_per_lot_per_night,
            direction=self.position,
        )
        swap_r_signed = swap_usd_amt / risk_usd if risk_usd else 0.0

        net_r = gross_r - spread_cost_r - commission_cost_r + swap_r_signed

        self.total_spread_cost_usd += spread_cost_usd_amt
        self.total_commission_usd += commission_usd_amt
        self.total_swap_usd += swap_usd_amt

        self.balance += net_r * risk_usd

        self.trades.append({
            "type": "long" if self.position == 1 else "short",
            "entry": self.entry_price,
            "exit": price,
            "R": net_r,
            "gross_R": gross_r,
            "spread_cost_R": spread_cost_r,
            "commission_cost_R": commission_cost_r,
            "swap_cost_R": swap_r_signed,
            "nights_held": held_nights,
            "bars": i,
            "status": status,
        })

        self.position = 0
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.risk = None
        self.bars_open = 0
        self._entry_spread_price = None
        self._entry_time = None

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

        # Gross (cost-free) profit factor, for comparison against the net
        # figure above -- the gap between the two IS your total cost drag.
        gross_wins = [t["gross_R"] for t in self.trades if t["gross_R"] > 0]
        gross_losses = [t["gross_R"] for t in self.trades if t["gross_R"] < 0]
        gross_total_win = sum(gross_wins)
        gross_total_loss = abs(sum(gross_losses))
        gross_profit_factor = (
            (gross_total_win / gross_total_loss) if gross_total_loss
            else (float("inf") if gross_total_win > 0 else 0.0)
        )

        return {
            "final_balance": self.balance,
            "total_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0,
            "avg_r": avg_r,
            "avg_win_r": avg_win,
            "avg_loss_r": avg_loss,
            "profit_factor": profit_factor,
            "gross_profit_factor": gross_profit_factor,
            "expectancy_r": expectancy,
            "total_spread_cost_usd": self.total_spread_cost_usd,
            "total_commission_usd": self.total_commission_usd,
            "total_swap_usd": self.total_swap_usd,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }
