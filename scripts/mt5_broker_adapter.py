from datetime import datetime, timezone


class MT5UnavailableError(RuntimeError):
    pass


class MT5BrokerAdapter:

    def __init__(self, magic_number=26072026):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:
            raise MT5UnavailableError(
                "MetaTrader5 package is not installed in this environment."
            ) from exc

        self.mt5 = mt5
        self.magic_number = magic_number

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

    def symbol_info(self, symbol):
        return self.mt5.symbol_info(symbol)

    def normalize_volume(self, symbol, volume):
        info = self.symbol_info(symbol)
        if info is None:
            return volume

        min_volume = getattr(info, "volume_min", 0.01) or 0.01
        max_volume = getattr(info, "volume_max", volume) or volume
        step = getattr(info, "volume_step", 0.01) or 0.01
        clipped = max(min_volume, min(volume, max_volume))
        steps = round(clipped / step)
        return max(min_volume, round(steps * step, 8))

    def order_calc_margin(self, direction, symbol, volume, price):
        order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL
        return self.mt5.order_calc_margin(order_type, symbol, volume, price)

    def history_deals_since(self, since_time, symbol=None, magic=None):
        deals = self.mt5.history_deals_get(since_time, datetime.now(timezone.utc))
        if deals is None:
            return []

        filtered = []
        for deal in deals:
            if symbol is not None and getattr(deal, "symbol", None) != symbol:
                continue
            if magic is not None and getattr(deal, "magic", None) != magic:
                continue
            filtered.append(deal)
        return filtered

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
            "magic": self.magic_number,
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        return self.mt5.order_send(request)

    def modify_position(self, ticket, symbol, stop_loss=None, take_profit=None):
        position = next((p for p in self.mt5.positions_get(symbol=symbol) or [] if p.ticket == ticket), None)
        if position is None:
            return None

        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": stop_loss if stop_loss is not None else position.sl,
            "tp": take_profit if take_profit is not None else position.tp,
            "magic": self.magic_number,
            "comment": "QuantFX manage",
        }
        return self.mt5.order_send(request)

    def close_position(self, position):
        symbol = position.symbol
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        is_buy = position.type == self.mt5.POSITION_TYPE_BUY
        price = tick.bid if is_buy else tick.ask
        order_type = self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": position.ticket,
            "volume": position.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "QuantFX close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        return self.mt5.order_send(request)
