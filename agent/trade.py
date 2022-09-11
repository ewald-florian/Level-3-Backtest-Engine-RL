# TODO: What exactly do I need Trade for?
# - Trade could also just use Order and filter for filled orders...
# - Or I call Trade inside the MarketState _update_with_agent_message part...

# Store executed agent trades

class TradeLog:

    history = list()  # instance store

    def __init__(self, timestamp, market_id, side, quantity, price):
        """
        ...
        """

        # static attributes from arguments
        self.timestamp = timestamp
        self.market_id = market_id
        self.side = side
        self.quantity = quantity
        self.price = price
        self.trade_id = len(self.__class__.history)
        # combo trades
        # agent_msg_num (order ref)
        # tc/commissions

        #

        # ...
        print("(INFO) trade {trade_id} was executed: {self}".format(
            trade_id=self.trade_id,
            self=self,
        ))

        # global attributes update
        self.__class__.history.append(self)

    def __str__(self):
        """
        String representation.
        """

        string = "{side} {market_id} with {quantity}@{price}, {time}".format(
            time=self.timestamp,
            market_id=self.market_id,
            side=self.side,
            quantity=self.quantity,
            price=self.price,
        )

        return string

    @classmethod
    def reset_history(class_reference):
        """
        Reset trade history.
        """
        del class_reference.history[:]