import MetaTrader5 as mt5



class Execution:


    def __init__(
        self,
        magic,
        risk_pct
    ):

        self.magic = magic
        self.risk_pct = risk_pct



    def calculate_lot(
        self,
        symbol,
        equity,
        stop_distance
    ):

        info = mt5.symbol_info(symbol)

        if info is None:
            return 0


        risk_money = (
            equity *
            self.risk_pct /
            100
        )


        points = (
            stop_distance /
            info.point
        )


        if points <= 0:
            return info.volume_min


        lot = (
            risk_money /
            (
                points *
                info.trade_tick_value
            )
        )


        lot = max(
            info.volume_min,
            min(
                lot,
                info.volume_max
            )
        )


        return round(
            lot,
            2
        )




    def send_order(
        self,
        symbol,
        direction,
        lot,
        sl,
        tp
    ):


        tick = mt5.symbol_info_tick(symbol)


        if tick is None:
            return False



        if direction == 1:

            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask


        else:

            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid



        request = {

            "action":
                mt5.TRADE_ACTION_DEAL,

            "symbol":
                symbol,

            "volume":
                lot,

            "type":
                order_type,

            "price":
                price,

            "sl":
                sl,

            "tp":
                tp,

            "magic":
                self.magic,

            "comment":
                "VX PROP V2",

            "type_time":
                mt5.ORDER_TIME_GTC,

            "type_filling":
                mt5.ORDER_FILLING_IOC
        }



        result = mt5.order_send(request)



        if result.retcode != mt5.TRADE_RETCODE_DONE:

            print(
                "ORDER FAILED:",
                result.comment
            )

            return False



        return True