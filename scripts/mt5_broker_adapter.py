class MT5UnavailableError(RuntimeError):
    pass


class MT5BrokerAdapter:

    def __init__(self):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:
            raise MT5UnavailableError(
                "MetaTrader5 package is not installed in this environment."
            ) from exc

        self.mt5 = mt5

    def initialize(self):
        if not self.mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {self.mt5.last_error()}")

    def shutdown(self):
        self.mt5.shutdown()

    def account_equity(self):
        info = self.mt5.account_info()
        return info.equity if info else None

    def positions_total(self, symbol=None):
        positions = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        return 0 if positions is None else len(positions)

    def rates_copy(self, symbol, timeframe, count=500):
        return self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    def place_order(self, *, symbol, direction, volume, stop_loss, take_profit, comment="QuantFX"):
        order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL
        tick = self.mt5.symbol_info_tick(symbol)
        price = tick.ask if direction == 1 else tick.bid

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 26072026,
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        return self.mt5.order_send(request)
