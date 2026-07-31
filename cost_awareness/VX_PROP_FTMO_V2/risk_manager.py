from datetime import datetime


class RiskManager:

    def __init__(
        self,
        start_equity,
        daily_halt_pct,
        total_halt_pct,
        profit_target
    ):

        self.start_equity = start_equity
        self.daily_halt_pct = daily_halt_pct
        self.total_halt_pct = total_halt_pct
        self.profit_target = profit_target

        self.day = None
        self.day_start_equity = start_equity

        self.daily_halted = False
        self.total_halted = False
        self.passed = False


    def update(self, equity):

        today = datetime.utcnow().date()

        if self.day != today:

            self.day = today
            self.day_start_equity = equity
            self.daily_halted = False


        total_dd = (
            (self.start_equity - equity)
            /
            self.start_equity
            *
            100
        )


        daily_dd = (
            (self.day_start_equity - equity)
            /
            self.day_start_equity
            *
            100
        )


        profit = (
            (equity - self.start_equity)
            /
            self.start_equity
            *
            100
        )


        if total_dd >= self.total_halt_pct:
            self.total_halted = True


        if daily_dd >= self.daily_halt_pct:
            self.daily_halted = True


        if profit >= self.profit_target:
            self.passed = True


        return {
            "total_dd": total_dd,
            "daily_dd": daily_dd,
            "profit": profit
        }


    def can_trade(self):

        if self.total_halted:
            return False

        if self.daily_halted:
            return False

        if self.passed:
            return False

        return True