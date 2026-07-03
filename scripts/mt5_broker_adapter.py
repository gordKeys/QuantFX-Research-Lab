from datetime import datetime, timezone


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

    def positions_get(self, symbol=None):
        return self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()

    def symbol_info(self, symbol):
        return self.mt5.symbol_info(symbol)

    def rates_copy(self, symbol, timeframe, count=500):
        return self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    def order_calc_margin(self, direction, symbol, volume, price):
        order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL
        return self.mt5.order_calc_margin(order_type, symbol, volume, price)

    def normalize_volume(self, symbol, volume):
        info = self.symbol_info(symbol)
        if info is None:
            return volume

        min_volume = getattr(info, "volume_min", 0.01) or 0.01
        max_volume = getattr(info, "volume_max", volume) or volume
        step = getattr(info, "volume_step", 0.01) or 0.01

        clipped = max(min_volume, min(volume, max_volume))
        steps = round(clipped / step)
        normalized = steps * step
        return max(min_volume, round(normalized, 8))

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
            "magic": 26072026,
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        return self.mt5.order_send(request)

    def modify_position(self, ticket, symbol, stop_loss, take_profit):
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": 26072026,
            "comment": "QuantFX manage",
        }
        return self.mt5.order_send(request)

    def close_position(self, ticket, symbol, volume, direction):
        order_type = self.mt5.ORDER_TYPE_SELL if direction == 1 else self.mt5.ORDER_TYPE_BUY
        tick = self.mt5.symbol_info_tick(symbol)
        price = tick.bid if direction == 1 else tick.ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 26072026,
            "comment": "QuantFX close",
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
            "magic": 26072026,
            "comment": "QuantFX",
        }
        return self.mt5.order_send(request)

    def close_position(self, position):
        symbol = position.symbol
        volume = position.volume
        direction = position.type
        tick = self.mt5.symbol_info_tick(symbol)
        price = tick.bid if direction == self.mt5.POSITION_TYPE_BUY else tick.ask
        order_type = self.mt5.ORDER_TYPE_SELL if direction == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": position.ticket,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 26072026,
            "comment": "QuantFX profit lock",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        return self.mt5.order_send(request)
