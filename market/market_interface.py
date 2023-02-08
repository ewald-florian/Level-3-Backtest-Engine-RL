#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
"""
Market Interface Module for the Level-3 backtest engine.
Interface to submit, modify and cancel orders.
"""



import numpy as np

from market.market import Market

# TODO: _assert_exposure_budget -> needs OMS / AgentMetrics (filter unexecuted
#   trades etc.)
# TODO: long-only (-> AgentMetrics), short-only(AgentMetrics), check if current
#  exposure is negative or positive...


class MarketInterface:
    """
    MarketInterface allows to submit new orders into Market, Cancel
    existing orders and the Modification of existing orders.  There are
    two different types of submittable agent messages:

    1.) Simulated Messages:
    -----------------------
    - Are stored and managed in OrderManagementSytem and seperated
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

    Prices and Quantities
    ---------------------
    Prices have to be entered with 8 decimals (without comma), quantities
    have to be entered with 4 decimals (without comma). For example, a Euro
    Price of 98,55€ must be entered as 9855000000. A quantity of 17 as
    170000.

    OrderManagementSystem
    ---------------------
    All agent messages are stored and maintained in the OrderManagementSystem
    order_list which is a class attribute of OMS.

    Template_IDs:
    -------------
    The major part of the agent order management is done via template_ids:
        99999 = active order (buy or sell)
        66666 = cancellation message
        44444 = modification message
        33333 = cancelled order
        11111 = executed order

    Matching:
    ---------
    Agent orders can be matched against 1) the limit order book and 2) trade
    executions which simulates the execution against incoming market- or
    marketable orders. When there are active agent orders in the OMS,
    the matching mechanism has to be called each time, the internal state
    changes. This has to be specified in the Market method
    update_simulation_with_exchange_message. For simulated agent orders, call
    _simulate_agent_order_matching(), for impact agent orders, call
    _impact_agent_order_matching(). Impact orders and simulated orders have
    to be matched separately.
    """

    def __init__(self,
                 tc_factor: float = 0,
                 exposure_limit: float = None,
                 long_only: bool = None,
                 short_only: bool = None):
        """
        Initiate MarketInterface. Tc-factors defines the transactions costs
        in basis points of the traded volume. Exposure limit can be used to
        limit the total exposure an agent is allowed to place. Long-only and
        short-only can be used to allow only for net long and net short
        positions (for example, an long-only agent is still allowed to sell
        but only a quantity up to his current inventory.)

        :param tc_factor
            float, transaction costs in bps (100th of %)
        :param exposure_limit
            int, agent exposure limit, blocks additional submissions
        :param long_only
            bool, define long-only agent
        :param short_only
            bool, short-only agent
        """

        # static attributes from arguments
        self.tc_factor = tc_factor
        self.exposure_limit = exposure_limit
        self.long_only = long_only
        self.short_only = short_only

    # 1.) without order book impact (simulation) . . . . . . . . . . . . . .

    def submit_order(self, side: int, quantity: int, limit: int = None):
        """
        Submit a simulated order which does not affect the internal state of
        the market (no market impact). Limit should consider the market
        ticksize, not permitted limits will cause an error. Active orders
        are market with a template_id of 99999.

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
    def cancel_order(order_message_id: int):
        """
        Cancel a simulated order. Message can be identified by message_id.
        Cancelled orders are marked with a template_id of 33333 in the
        OMS.order_list.

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
        Modify an existing simulation order by changing its price to new_price
        or its quantity to new_quantity. The order to be modified can be
        identified my its message_id Note that changes in prices and
        increases in quantity will lead to a new inferior priority time. Sole
        decreases in quantity maintain the original priority time.
        -----
        Zertifizierter Börsenhändler Kassamarkt Handbuch (Seite 168):

        'Eine Orderänderung führt immer dann zu einer neuen Zeitpriorität der
        Order, wenn entweder das Limit geändert wird oder die Orderänderung
        einen nachteiligen Einfluss auf die Priorität der Ausführung anderer
        Orders im Orderbuch hätte (z. B. Erhöhung des Volumens einer
        bestehenden Order). Sollte hingegen das Volumen einer bestehenden Order
        verkleinert werden, so bleibt die ursprüngliche Zeitpriorität
        erhalten.'
        -----
        :param order_message_id
            int, message_id of order to modify
        :param new_price
            int, new price
        :param new_quantity
            int, new quantity
        """
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
                            limit: int = None):
        """
        Submit order with market impact. This order will be placed in the
        internal state of the market but also added to the regular
        OMS.order_list. Impact messages are marked with an "impact_flag" which
        is set to 1 (inside Market).
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

        # submit directly to MarketStateAttribute.instance
        Market.instances["ID"].update_with_agent_message_impact(message)

    @staticmethod
    def cancel_order_impact(order_message_id: int):
        """
        Cancel impact order. This will mark the impact order as cancelled
        by changing its template_id to 33333 and remove the order from the
        internal state. The order to be cancelled must be identified by its
        message_id.

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
        Modify an existing impact order by changing its price to new_price
        or its quantity to new_quantity. The order to be modified can be
        identified my its message_id Note that changes in prices and
        increases in quantity will lead to a new inferior priority time. Sole
        decreases in quantity maintain the original priority time. The order
        will be modified in the OMS as well as in the internal state.

        :param order_message_id
            int, message_id of order to modify
        :param new_price
            int, new price
        :param new_quantity
            int, new quantity
        """
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
        Market.instances["ID"].update_with_agent_message_impact(message)

    # TODO: implement
    def _assert_exposure_budget(self):
        pass

    def _assert_long_only(self):
        pass

    def _assert_short_only(self):
        pass

    def _estimated_tc(self, limit, quantity):
        """
        Estimated tc in Euro with 8 decimals (without comma). Same format than
        prices. Example: 40_00000000 = 40€.
        """
        tc = limit*quantity*self.tc_factor*1e-4
        return tc

    def reset(self):
        """
        Reset MarketInterface.
        """
        # call init with static attributes as arguments
        self.__init__(tc_factor=self.tc_factor,
                      exposure_limit=self.exposure_limit,
                      long_only=self.long_only,
                      short_only=self.short_only,
                      )
