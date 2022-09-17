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


from market.market import Market
#from agent.agent_order import OrderManagementSystem as OMS

class MarketInterface:

    def __init__(self,
                 tc_factor:float=0,
                 exposure_limit:float=100_000
                 ):

        # static attributes from argumens
        self.latency = latency
        self.tc_factor = tc_factor
        self.exposure_limit = exposure_limit

    # without order book impact (simulation) . . . . . . . . . . . . . . . . .

    @staticmethod
    def submit_order(side, quantity, limit=None):
        """
        For simulated orders which do not affect the market state.
        Send submission order.
        :param side:
            1(Buy), 2(Sell)
        :param quantity:
            int,
        :param limit:
            int, limit price
        """

        message = dict()
        # TemplateID 99999 for submission
        message['template_id'] = 99999
        message["side"] = side
        message["price"] = limit
        message["quantity"] = quantity

        # submit to Market
        Market.instances["ID"].update_simulation_with_agent_message(message)

    @staticmethod
    def cancel_order(order_message_id):
        """
        For simulated orders which do not affect the market state.
        Send cancellation order. Message can be identified by message_id.

        :param order_message_id
            int, unique agent message identifier
        """
        #TODO: assert if order to be cancelled exists... (sort OMS)

        message = dict()
        # TemplateID 66666 for cancellation
        message['template_id'] = 66666
        # message number of the order to be cancelled
        message['order_message_id'] = order_message_id

        # send cancellation message to Market
        Market.instances["ID"].update_simulation_with_agent_message(message)

    @staticmethod
    def modify_order(self):
        """
        ...
        """
        raise NotImplementedError ("order modify not implemented yet")

    # with order book impact . . . . . . . . . . . . . . . . . . . . . .

    @staticmethod
    def submit_order_impact(side, quantity, timestamp, limit=None):
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
        Market.instances["ID"].update_with_agent_message_impact(message)

    # TODO: Überarbeiten, aktuell nicht kompatibel...
    @staticmethod
    def cancel_order_impact(order_message_id):
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
        # message number of the order to be cancelled
        message['order_message_id'] = order_message_id

        # send cancellation message to MarketState
        Market.instances["ID"].update_with_agent_message_impact(message)

    @staticmethod
    def modify_order_impact(self):
        """
        ...
        """
        raise NotImplementedError ("order modify not implemented yet")

    # statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

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