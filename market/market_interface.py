#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 06/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Market Interface Module for the Level-3 backtest engine.
Interface to submit and cancel orders.
"""
# ---------------------------------------------------------------------------

# TODO: implement assertions (ca. 20 min.)

import numpy as np

from market.market import Market
from agent.agent_order import OrderManagementSystem as OMS
# from agent.agent_order import OrderManagementSystem as OMS


class MarketInterface:

    def __init__(self,
                 tc_factor: float = 0,
                 exposure_limit: float = None,
                 long_only: bool = None,
                short_only: bool = None):
        """
        MarketInterface allows to submit new orders into Market or
        cancel existing orders. There are two separate types of submittable
        agent messages:

        1.) Simulated Messages:
        -----------------------
        - Are stored and managed in OrderManagementSytem and  seperated
        from the actual market state.
        - Are matched against historical limit and/or marekt orders in
        a simulated matching process.
        - Do not affect the internal market state and hence do not have
        "real" market impact.
        - Instead, market impact can be artificially generated as a part
        of the backtest simulation.

        2.) Impact Messages:
        --------------------
        - Impact submissions place orders into the internal market state.
        - When impact orders can be matched, their counter-orders get executed
        and vanish from the orderbook.
        - Impact submissions do have lasting market impact.
        - Note, that the orderbook can not recover in a realistic way from
        excessive agent impact executions.

        Note: "Market Impact" here refers to its  broader definition of
        affecting the market by changing the internal market state. It does
        not necessarily mean an immediate price impact on the midpoint.

        Typically, a backtest would only use either simulated ore impact agent
        orders. But technically, it is possible to mix them.

        :param tc_factor
            float, transaction costs in bps (100th of %)
        :param exposure_limit
            int, agent exposure limit, blocks additional submissions
        """

        # static attributes from arguments
        self.tc_factor = tc_factor
        self.exposure_limit = exposure_limit

        #TODO: get ticksize
        #self.ticksize = Market.instances['ID'].ticksize

    # 1.) without order book impact (simulation) . . . . . . . . . . . . . .

    def submit_order(self, side: int, quantity: int, limit: int = None):
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
        #TODO: if exposure limit, assert if exposure budget left...
        #TODO: Assert if price is in ticksize etc.
        # Assert correct format
        # long/short (depends on current position if is allowed to sell,
        # e.g. long only can only sell if position_value > 0 and > estimated
        # sell volume

        #self._assert_price_permitted(limit)

        message = dict()
        # TemplateID 99999 for submission
        message['template_id'] = 99999
        message["side"] = side
        message["price"] = limit
        message["quantity"] = quantity

        # submit to Market
        Market.instances["ID"].update_simulation_with_agent_message(message)

    @staticmethod
    def cancel_order(order_message_id: int):
        """
        For simulated orders which do not affect the market state.
        Send cancellation order. Message can be identified by message_id.

        :param order_message_id
            int, unique agent message identifier of order to be cancelled
        """
        # TODO: assert if order to be cancelled exists... (sort OMS)

        message = dict()
        # TemplateID 66666 for cancellation
        message['template_id'] = 66666
        # message number of the order to be cancelled
        message['order_message_id'] = order_message_id

        # send cancellation message to Market
        Market.instances["ID"].update_simulation_with_agent_message(message)

    def modify_order(self,
                     order_message_id,
                     new_price=None,
                     new_quantity=None):
        """
        template_id = 444444
        -----
        Zertifizierter Börsenhändler Kassamarkt Hnadbuch (Seit 168):

        'Eine Orderänderung führt immer dann zu einer neuen Zeitpriorität der
        Order, wenn entweder das Limit geändert wird oder die Orderänderung
        einen nachteiligen Einfluss auf die Priorität der Ausführung anderer
        Orders im Orderbuch hätte (z. B. Erhöhung des Volumens einer
        bestehenden Order). Sollte hingegen das Volumen einer bestehenden Order
        verkleinert werden, so bleibt die ursprüngliche Zeitpriorität
        erhalten.' #todo: Stelle im offiziellen Release raussuchen.
        -----
        :param order_message_id
            int, message_id of order to modify
        :param new_price
            int, new price
        :param new_quantity
            int, new quantity
        """
        if new_price:
            self._assert_price_permitted(new_price)

        message = dict()
        # TemplateID 66666 for cancellation
        message['template_id'] = 44444
        # message number of the order to be cancelled
        message['order_message_id'] = order_message_id

        if new_price:
            message['new_price'] = new_price

        if new_quantity:
            message['new_quantity'] = new_quantity

        # send modification message to Market
        Market.instances["ID"].update_simulation_with_agent_message(message)

    # 2.) with order book impact . . . . . . . . . . . . . . . . . . . . . .

    def submit_order_impact(self,
                            side: int,
                            quantity: int,
                            timestamp: int,
                            limit: int = None):
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

        self._assert_price_permitted(limit)

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
    def cancel_order_impact(order_message_id: int):
        """
        Send cancellation order. Requires price, limit and timestamp of order
        which should be cancelled.
        :param order_message_id
            int, id of message to be cancelled
        """
        message = dict()
        # TemplateID 66666 for cancellation
        message['template_id'] = 66666
        # message number of the order to be cancelled
        message['order_message_id'] = order_message_id

        # send cancellation message to MarketState
        Market.instances["ID"].update_with_agent_message_impact(message)

    def modify_order_impact(self,
                            order_message_id: int,
                            new_price: int,
                            new_quantity: int):
        """
        Modilfy impact order.

        :param order_message_id
            int, message_id of order to modify
        :param new_price
            int, new price
        :param new_quantity
            int, new quantity
        """
        if new_price:
            self._assert_price_permitted(new_price)

        message = dict()
        # TemplateID 66666 for cancellation
        message['template_id'] = 44444
        # message number of the order to be cancelled
        message['order_message_id'] = order_message_id

        if new_price:
            message['new_price'] = new_price

        if new_quantity:
            message['new_quantity'] = new_quantity

        # TODO: implement processing of impact message
        # send modification message to Market
        Market.instances["ID"].update_with_agent_message_impact(message)

    #TODO
    def _assert_price_permitted(self, price: int):
        """
        Check if price respects the permitted tick size.
        """
        return np.gcd(self.ticksize, price) == self.ticksize

    #TODO
    def _assert_message_exists(self, order_message_id):
        """
        Assert of order exists and is active (template_id 99999)
        :param order_message_id
            int, message_id of order to be cancelled

        """

       #  [d for d in OMS.order_list if d['message_id'] == order_message_id]
        pass
    #TODO
    def _assert_exposure_budget(self):
        pass

    def reset(self):
        """
        Reset class
        """
        self.__init__()
