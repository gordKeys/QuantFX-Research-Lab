from engine.trade import Trade


class ExecutionEngine:

    def __init__(self):

        self.open_trade = None

    def has_position(self):

        return self.open_trade is not None

    def open_position(self, direction, entry, stop, target, size, strategy, regime, time):

        # simulate spread
        spread = 0.0001 if direction == 1 else -0.0001
        entry = entry + spread

        self.open_trade = Trade(
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            position_size=size,
            strategy=strategy,
            regime=regime,
            entry_time=time,
            max_favorable_price=entry,
            max_favorable_pnl=0.0,
            bars_open=0,
        )

    def manage_open_trade(self, current_price, current_time, breakeven_at_r=0.2, trail_at_r=0.4, trail_buffer_r=0.25, max_bars=48, profit_fade_pct=0.35, profit_floor_r=0.25):
        if self.open_trade is None:
            return None

        t = self.open_trade
        t.bars_open += 1
        risk = abs(t.entry_price - t.stop_loss)
        if risk <= 0:
            return None

        current_pnl = (current_price - t.entry_price) if t.direction == 1 else (t.entry_price - current_price)
        if current_pnl > t.max_favorable_pnl:
            t.max_favorable_pnl = current_pnl

        if t.max_favorable_pnl >= risk * profit_floor_r:
            if current_pnl <= t.max_favorable_pnl * (1 - profit_fade_pct):
                t.exit_price = current_price
                t.exit_time = current_time
                t.status = "PROFIT_FADE"
                t.pnl = current_pnl * t.position_size
                t.r_multiple = current_pnl / risk
                finished = self.open_trade
                self.open_trade = None
                return finished

        if t.direction == 1:
            t.max_favorable_price = max(t.max_favorable_price, current_price)
            open_r = (current_price - t.entry_price) / risk
            if open_r >= breakeven_at_r:
                t.stop_loss = max(t.stop_loss, t.entry_price)
            if open_r >= trail_at_r:
                trail_distance = risk * trail_buffer_r
                t.stop_loss = max(t.stop_loss, current_price - trail_distance)
        else:
            t.max_favorable_price = min(t.max_favorable_price, current_price)
            open_r = (t.entry_price - current_price) / risk
            if open_r >= breakeven_at_r:
                t.stop_loss = min(t.stop_loss, t.entry_price)
            if open_r >= trail_at_r:
                trail_distance = risk * trail_buffer_r
                t.stop_loss = min(t.stop_loss, current_price + trail_distance)

        if t.bars_open >= max_bars:
            t.exit_price = current_price
            t.exit_time = current_time
            t.status = "TIME_EXIT"
            pnl = (current_price - t.entry_price) if t.direction == 1 else (t.entry_price - current_price)
            t.pnl = pnl * t.position_size
            t.r_multiple = pnl / risk
            finished = self.open_trade
            self.open_trade = None
            return finished

    def update(

        self,

        high,

        low,

        current_time

    ):

        if self.open_trade is None:
            return None

        t = self.open_trade

        self.manage_open_trade(current_price=(high + low) / 2, current_time=current_time)
        if self.open_trade is None:
            return t

        if t.direction == 1:

            if low <= t.stop_loss:

                t.exit_price = t.stop_loss
                t.status = "STOP"

            elif high >= t.take_profit:

                t.exit_price = t.take_profit
                t.status = "TARGET"

            else:
                return None

        else:

            if high >= t.stop_loss:

                t.exit_price = t.stop_loss
                t.status = "STOP"

            elif low <= t.take_profit:

                t.exit_price = t.take_profit
                t.status = "TARGET"

            else:
                return None

        t.exit_time = current_time

        risk = abs(t.entry_price - t.stop_loss)

        reward = abs(t.exit_price - t.entry_price)

        if t.direction == 1:
            pnl = (t.exit_price - t.entry_price)
        else:
            pnl = (t.entry_price - t.exit_price)

        t.pnl = pnl * t.position_size

        t.r_multiple = reward / risk

        finished = self.open_trade

        self.open_trade = None

        return finished
