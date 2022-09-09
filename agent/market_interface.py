#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 06/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Market Interface Module for the Level-3 backtest engine.
Interface to submit and cancel orders.
"""
# ---------------------------------------------------------------------------

#TODO: handle circular import problplems...
from market.market_state_v1 import MarketStateAttribute
from agent.order import Order

class MarketInterface:

    def __init__(self,
                 latency:int=0,
                 tc_factor:float=0,
                 exposure_limit:float=100_000
                 ):

        # -- static attributes from argumens
        self.latency = latency
        self.tc_factor = tc_factor
        self.exposure_limit = exposure_limit

        # -- dynamic attributes
        # ...


    # without order book impact (simulation) . . . . . . . . . . . . . . . . .

    def submit_order(self, side, quantity, timestamp, limit=None):
        """
        For simulated orders which do not affect the market state.
        Send submission order.
        :param side:
            1(Buy), 2(Sell)
        :param quantity:
            int,
        :param timestamp:
            int, unix timestamp
        :param limit:
            int, limit price
        """
        # TODO: Assert if price is in ticksize etc.

        message = dict()
        # TemplateID 99999 for submission
        message['template_id'] = 99999
        message["side"] = side
        message["price"] = limit
        message["quantity"] = quantity
        message["timestamp"] = timestamp

        # submit to
        MarketStateAttribute.instance.simulated_update_with_agent_message(message)

        # append message to order-list
        Order(message)
        print(len(Order.order_list))

    def cancel_order(self, side, limit, timestamp):
        """
        For simulated orders which do not affect the market state.
        Send cancellation order. Requires price, limit and timestamp of order
        which should be cancelled.
        :param side
            1 (buy) or 2 (Sell)
        :param limit
            int, price
        :param timestamp
            int, unix timestamp of order which should be cancelled
        """
        #TODO: assert if order to be cancelled exists...

        message = dict()
        # TemplateID 66666 for cancellation
        message['template_id'] = 66666
        message["side"] = side
        message["price"] = limit
        # timestamp of order which should be cancelled
        message["timestamp"] = timestamp

        # send cancellation message to MarketState
        MarketStateAttribute.instance.simulated_update_with_agent_message(message)

        # TODO: Update order status to 'CANCELLED' in Order.history...

    def modify_order(self):
        pass

    # with order book impact . . . . . . . . . . . . . . . . . . . . . .

    def submit_order_impact(self, side, quantity, timestamp, limit=None):
        """
        Send submission order.
        :param side:
            1(Buy), 2(Sell)
        :param quantity:
            int,
        :param timestamp:
            int, unix timestamp
        :param limit:
            int, limit price
        """
        # TODO: Assert if price is in ticksize etc.

        message = dict()
        # TemplateID 99999 for submission
        message['template_id'] = 99999
        message["side"] = side
        message["price"] = limit
        message["quantity"] = quantity
        message["timestamp"] = timestamp

        # submit directly to MarketStateAttribute.instance
        MarketStateAttribute.instance.update_with_agent_message(message)

        # append message to order-list
        Order(message)
        print(len(Order.order_list))

    def cancel_order_impact(self, side, limit, timestamp):
        """
        Send cancellation order. Requires price, limit and timestamp of order
        which should be cancelled.
        :param side
            1 (buy) or 2 (Sell)
        :param limit
            int, price
        :param timestamp
            int, unix timestamp of order which should be cancelled
        """
        message = dict()
        # TemplateID 66666 for cancellation
        message['template_id'] = 66666
        message["side"] = side
        message["price"] = limit
        # timestamp of order which should be cancelled
        message["timestamp"] = timestamp

        # send cancellation message to MarketState
        MarketStateAttribute.instance.update_with_agent_message(message)

        # TODO: Update order status to 'CANCELLED' in Order.history...or whatever

    def modify_order_impact(self):
        pass

    def get_filtered_orders(self):
        pass

    def get_filtered_trades(self):
        pass

    @property
    def exposure(self):
        pass

    @property
    def pnl_realized(self):
        # for each round-trip seperately to use as reward,
        # alternatively, compute diff between old and new PnL realized
        pass

    @property
    def pnl_unrealized(self):
        # exposure * price difference since execution
        # consider spread?
        # consider midpoints?
        pass

    @property
    def exposure_budget_left(self):
        pass

    @property
    def transaction_costs(self):

        #trading_volume = price * quantity
        #transaction_costs = volume * self.tc_factor
        #return(round(transaction_costs, 4))
        pass