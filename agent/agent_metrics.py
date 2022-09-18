#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 16/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Agent Performance Metrics during Trading"""

# ---------------------------------------------------------------------------
import textwrap

from agent.agent_order import OrderManagementSystem as OMS
from agent.agent_trade import AgentTrade
from market.market import Market
from market.market_trade import MarketTrade

#TODO: is there a way to access properties withoz instantiating AgentMetrics? I can use
# staticmethods but many methods need self...

# TODO: testing, debugging, str, roundtrip-PnLs
class AgentMetrics:

    def __init__(self, tc_factor: float = 1e-3,
                 exposure_limit: int = 0):

        self.tc_factor = tc_factor
        self.exposure_limit = exposure_limit

    @staticmethod
    def get_filtered_messages(side=None, template_id=None):
        """
        Filter messages by side (1: Bid, 2: Ask) and template_id:
        - 99999 = active order
        - 66666 = cancellation message
        - 11111 = executed order
        - 33333 = cancelled order

        :param side
            int, 1 or 2
        :param template_id
            int, template_id of message
        """
        if side:
            assert side in [1, 2], "(AgentMetrics) Side " \
                                   "not valid, must be 1 or 2"

        if template_id:
            assert template_id in [99999, 66666, 11111, 33333], \
                "(AgentMetrics) template_id not valid"

        filtered_messages = OMS.order_list

        # orders must have requested market_id
        if side:
            filtered_messages = filter(lambda d: d['side'] == side,
                                       filtered_messages)
        # orders must have requested side
        if template_id:
            filtered_messages = filter(lambda d: d['template_id'] == template_id,
                                       filtered_messages)

        return list(filtered_messages)

    @staticmethod
    def get_filtered_trades(side):
        """
        Filter trades by side (1 Buy, 2 Sell)

        :param side
            int, 1 or 2
        """
        assert side in [1, 2], "(AgentMetrics) Side " \
                               "not valid, must be 1 or 2"

        filtered_trades = AgentTrade.history

        filtered_trades = filter(lambda d: d['agent_side'] == side,
                                 filtered_trades)

        return list(filtered_trades)

    @property
    def unrealized_quantity(self):
        """
        Quantity of assets in open trades.
        """
        unrealized_quantity = 0

        for trade in AgentTrade.history:

            quantity = trade['executed_volume'] * 1e-4
            side = trade['agent_side']

            if side == 1:
                unrealized_quantity += quantity
            elif side == 2:
                unrealized_quantity -= quantity

        return unrealized_quantity

    @property
    def realized_quantity(self):
        # realized quantity
        realized_trades = self.get_realized_trades
        unrealized_trades = self.get_unrealized_trades

        realized_buy_trades = list(filter(lambda d: d['agent_side'] == 1, realized_trades))
        realized_sell_trades = list(filter(lambda d: d['agent_side'] == 2, realized_trades))
        realized_buy_quantity = sum(trade['executed_volume'] for trade in realized_buy_trades)
        realized_sell_quantity = sum(trade['executed_volume'] for trade in realized_sell_trades)

        realized_quantity = realized_buy_quantity - realized_sell_quantity

        return realized_quantity * 1e-4

    @property
    def position_value(self):
        """
        Current value of unrealized position, approximated with best-bid
        or best-ask value.
        """
        unrealized_quantity = self.unrealized_quantity
        if unrealized_quantity > 0:
            position_value = unrealized_quantity * \
                             Market.instances['ID'].best_bid

        elif unrealized_quantity < 0:
            position_value = unrealized_quantity * \
                             Market.instances['ID'].best_ask

        else:
            position_value = 0

        return position_value

    @property
    def exposure(self):
        """
        Current Exposure given in EUR. Based on entry value of
        unrealized trades.

        :return expsoure
            float, current exposure in EUR
        """
        #TODO: change: exposure based on buy-in price or current marekt price -> market price
        exposure = 0

        for trade in AgentTrade.history:

            side = trade['agent_side']
            price = trade['execution_price'] * 1e-8
            quantity = trade['executed_volume'] * 1e-4
            turnover = price * quantity

            if side == 1:
                exposure += turnover
            elif side == 2:
                exposure -= turnover

        return exposure

    @property
    def cash_position(self):
        """
        Current Cash Position.
        """
        cash_position = 0

        for trade in AgentTrade.history:

            side = trade['agent_side']
            price = trade['execution_price'] * 1e-8
            quantity = trade['executed_volume'] * 1e-4
            turnover = price * quantity

            if side == 1:
                cash_position -= turnover
            elif side == 2:
                cash_position += turnover

        return cash_position

    @property
    def get_unrealized_trades(self):
        """
        Currently unrealized trades.
        :return active_trades,
            list, contains active trades as dicts
        """

        unrealized_trades = []

        # 'long'
        if self.exposure > 0:
            buy_trades = list(filter(lambda d: d['agent_side'] == 1, AgentTrade.history))
            # sort by timestamp descending -> latest trade in the beginning of the list
            sorted_trades = sorted(buy_trades, key=lambda d: d['execution_time'])

        # 'short'
        elif self.exposure < 0:
            sell_trades = list(filter(lambda d: d['agent_side'] == 2, AgentTrade.history))
            sorted_trades = sorted(sell_trades, key=lambda d: d['execution_time'])

        else:
            pass  # 'neutral' / exposure = 0 / no unrealized trades

        # to deduct exposure of each trade from abs_exposure_split
        abs_exposure_split = abs(self.exposure)

        while abs_exposure_split > 0:

            for trade in sorted_trades:

                if abs_exposure_split > 0:
                    unrealized_trades.append(trade)

                    price = trade['execution_price'] * 1e-8
                    quantity = trade['executed_volume'] * 1e-4
                    turnover = price * quantity

                    abs_exposure_split -= turnover

        return unrealized_trades

    @property
    def get_realized_trades(self):
        """
        List of realized trades.
        """
        # TODO: checken was passiert wenn listen unvollständig sind...(am anfang)
        realized_trades = AgentTrade.history.copy()
        unrealized_trades = self.get_unrealized_trades

        if realized_trades:
            for trade in unrealized_trades:
                realized_trades.remove(trade)

            return realized_trades

    @property
    def vwap_buy(self):
        """
        Vwap buy.
        """
        # TODO: should the prices of unrealized trades be included?
        realized_trades = self.get_realized_trades

        realized_buy_trades = list(filter(lambda d: d['agent_side'] == 1, realized_trades))
        if realized_buy_trades:
            vwap_buy = sum(trade['execution_price'] * trade['executed_volume'] for trade in
                           realized_buy_trades) / sum(trade['executed_volume']
                                                      for trade in realized_buy_trades) * 1e-8
            vwap_buy = round(vwap_buy, 2)
        else:
            vwap_buy = 0


        return vwap_buy

    @property
    def vwap_sell(self):
        """
        Vwap sell.
        """
        realized_trades = self.get_realized_trades

        realized_buy_trades = list(filter(lambda d: d['agent_side'] == 2, realized_trades))
        if realized_buy_trades:
            vwap_sell = sum(trade['execution_price'] * trade['executed_volume'] for trade in
                            realized_buy_trades) / sum(trade['executed_volume']
                                                       for trade in realized_buy_trades) * 1e-8
            vwap_sell = round(vwap_sell, 2)
        else:
            vwap_sell = None
        return vwap_sell

    @property
    def vwap_score(self):
        """
        Current vwap score of agent.
        :return
        """
        # realized quantity

        market_trades = MarketTrade.history

        if market_trades:

            vwap_symbol = (sum(trade['price']*trade['quantity']
                        for trade in market_trades) /
                        sum(trade['quantity'] for
                        trade in market_trades))*1e-8

            realized_trades = self.get_realized_trades
            vwap_buy = self.vwap_buy
            vwap_sell = self.vwap_sell

            # cannot be None
            if not vwap_buy:
                vwap_buy = 0
            if not vwap_sell:
                vwap_sell = 0

            realized_buy_trades = list(filter(lambda d: d['agent_side'] == 1, realized_trades))
            realized_sell_trades = list(filter(lambda d: d['agent_side'] == 2, realized_trades))

            qt_bought = sum(trade['executed_volume'] for trade in realized_buy_trades)
            qt_sold = sum(trade['executed_volume'] for trade in realized_sell_trades)

            vwap_score = qt_bought * (vwap_symbol - vwap_buy) + qt_sold * (vwap_sell - vwap_symbol)
        else:
            vwap_score = 0
        # TODO: scale?
        return vwap_score

    @property
    def pnl_realized(self):
        """
        Current realized PnL of agent.
        """
        realized_quantity = self.realized_quantity
        vwap_buy = self.vwap_buy
        vwap_sell = self.vwap_sell

        if realized_quantity:
            # net long
            if realized_quantity > 0:
                pnl_realized = realized_quantity * (vwap_sell - vwap_buy)
            # net short
            elif realized_quantity < 0:
                pnl_realized = realized_quantity * (vwap_buy - vwap_sell)
            # no realized quantity
            else:
                pnl_realized = 0
        else:
            pnl_realized = 0

        return round(pnl_realized, 2)

    @property
    def pnl_unrealized(self):
        """
        Unrealized PnL.
        """
        unrealized_trades = self.get_unrealized_trades
        unrealized_quantity = self.unrealized_quantity

        if unrealized_trades:
                # average share price of unrealized trades
                asp = sum(trade['execution_price'] * trade['executed_volume'] for
                          trade in unrealized_trades) / sum(trade['executed_volume']
                                                            for trade in unrealized_trades)

                # net long position -> sell for current best-bid
                if unrealized_quantity > 0:
                    immediate_price = Market.instances['ID'].best_bid

                # net short position -> buy back for current best ask
                elif unrealized_quantity < 0:
                    immediate_price = Market.instances['ID'].best_ask

                pnl_unreal = unrealized_quantity * (immediate_price - asp) * 1e-8 * 1e-4
        else:
            pnl_unreal = 0

        return round(pnl_unreal, 2)

    @property
    def exposure_budget_left(self):
        """
        Remaining exposure budget in case on limited exposure.
        """
        return self.exposure - self.exposure_limit

    @property
    def transaction_costs(self):
        """
        Cumulated transaction costs of all trades.
        """
        trading_volume = sum(trade['execution_price'] * trade['executed_volume']
                             for trade in AgentTrade.history) * 1e-8 * 1e-4
        trading_volume = round(trading_volume * 1e-8 * 1e-4, 2)

        return trading_volume * self.tc_factor

    # TODO: adjust str
    def __str__(self):
        """
        String representation.
        """
        # get current timestamp from market
        timestamp = Market.instances['ID'].timestamp
        dt_timestamp = Market.instances['ID'].timestamp_datetime

        # string representation
        string = f"""
        ----------------------------
        timestamp:      {timestamp}
        datetime:       {dt_timestamp}
        ---
        exposure:       {self.exposure}
        pnl_realized:   {self.pnl_realized}
        pnl_unrealized: {self.pnl_unrealized}
        ----------------------------
        """
        return textwrap.dedent(string)
