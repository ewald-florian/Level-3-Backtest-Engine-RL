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
import copy


from agent.agent_order import OrderManagementSystem as OMS
from agent.agent_trade import AgentTrade
from market.market import Market
from market.market_trade import MarketTrade
from context.agent_context import AgentContext


# TODO: testing, debugging, str, roundtrip-PnLs


class AgentMetrics:

    def __init__(self,
                 tc_factor: float = 1e-3,
                 exposure_limit: int = 0):

        # static arguments
        self.tc_factor = tc_factor
        self.exposure_limit = exposure_limit

        # dynamic attributes
        # TODO: check if there is a better solution (maybe better in Reward...)
        self.last_number_of_trades = 0

    def get_filtered_messages(self, side=None, template_id=None):
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
            assert template_id in [99999, 66666, 11111, 33333, 44444], \
                "(AgentMetrics) template_id not valid"

        filtered_messages = OMS.order_list

        if filtered_messages:
            # orders must have requested market_id
            if side:
                filtered_messages = filter(lambda d: d['side'] == side,
                                           filtered_messages)

            # orders must have requested side
            if template_id:
                filtered_messages = filter(lambda d: d['template_id'] ==
                                            template_id, filtered_messages)

            return list(filtered_messages)

    def get_filtered_trades(self, side):
        """
        Filter trades by side (1 Buy, 2 Sell)

        :param side
            int, 1 or 2
        """
        assert side in [1, 2], "(AgentMetrics) Side " \
                               "not valid, must be 1 or 2"

        if AgentTrade.history:

            filtered_trades = AgentTrade.history

            filtered_trades = filter(lambda d: d['agent_side'] == side,
                                     filtered_trades)

            return list(filtered_trades)

    @property
    def time_since_last_submission(self):
        """Time since last order in nanoseconds."""
        time_since_last_order = 0.0000001
        if len(OMS.order_list) > 0:
            last_order_timestamp = OMS.order_list[-1]['timestamp']
            current_time = Market.instances["ID"].timestamp
            time_since_last_order = current_time - last_order_timestamp

        return time_since_last_order

    @property
    def time_since_last_trade(self):
        """Time since last trade in nanoseconds."""
        time_since_last_trade = 0.0000001
        if len(AgentTrade.history) > 0:
            # Execution time of latest trade.
            last_trade_timestamp = AgentTrade.history[-1]['execution_time']
            current_time = Market.instances["ID"].timestamp
            time_since_last_trade = current_time - last_trade_timestamp

        return time_since_last_trade

    @property
    def unrealized_quantity(self):
        """
        Quantity of assets in open trades. A negative realized quantity means
        the positions were net short.
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
    def executed_quantity(self):
        """Compute the executed (liquidated quantity)"""
        realized_quantity = sum(trade['executed_volume'] for trade in
                                AgentTrade.history)
        return realized_quantity

    @property
    def realized_quantity(self):
        """Compute the realized quantity."""

        realized_quantity = 0

        if len(AgentTrade.history) > 0:
            # realized quantity
            realized_trades = AgentTrade.history

            realized_buy_trades = list(filter(lambda d: d['agent_side'] == 1,
                                              realized_trades))

            realized_sell_trades = list(filter(lambda d: d['agent_side'] == 2,
                                               realized_trades))

            realized_buy_quantity = sum(trade['executed_volume'] for trade in
                                        realized_buy_trades)

            realized_sell_quantity = sum(trade['executed_volume'] for trade in
                                         realized_sell_trades)

            realized_quantity = realized_buy_quantity - realized_sell_quantity

        return realized_quantity * 1e-4

    @property
    def remaining_inventory(self, agent_side=2):
        """
        This metric is relevant for Optimal Execution Agents it computes
        the remaining inventory as the initial inventory minus the realized
        quantity
        :param agent_side
            int, side on which the OE agent trades. 2 for execution agent, 1
            for acquisition agent.
        """

        # Compute sum of quantity of respective side.
        realized_trades = AgentTrade.history
        if realized_trades:
            realized_trades = list(filter(lambda d: d['agent_side'] == agent_side,
                                          realized_trades))
            realized_quantity = sum(trade['executed_volume'] for trade in
                                         realized_trades)
            # Difference between initial inventory and remaining inventory.
            rem_inv = AgentContext.initial_inventory - realized_quantity
        else:
            rem_inv = AgentContext.initial_inventory
        return rem_inv

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
            buy_trades = list(filter(lambda d: d['agent_side'] == 1,
                                     AgentTrade.history))

            # sort by timestamp descending -> latest trade in the beginning
            # of the list
            sorted_trades = sorted(buy_trades, key=lambda d: d[
                'execution_time'])

        # 'short'
        elif self.exposure < 0:

            sell_trades = list(filter(lambda d: d['agent_side'] == 2,
                                      AgentTrade.history))
            sorted_trades = sorted(sell_trades, key=lambda d: d[
                'execution_time'])

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

    def get_realized_trades(self):
        """
        List of realized trades.
        """
        return AgentTrade.history

    @property
    def vwap_buy(self):
        """
        Vwap buy. Note: includes all trades, realized and not realized.
        """
        vwap_buy = None

        if AgentTrade.history:

            buy_trades = list(filter(lambda d: d['agent_side'] == 1,
                                              AgentTrade.history))
            if buy_trades:

                vwap_buy = sum(trade['execution_price'] *
                                trade['executed_volume']
                                for trade in buy_trades) / sum(
                                trade['executed_volume']
                                for trade in buy_trades) * 1e-8

                vwap_buy = round(vwap_buy, 4)

        return vwap_buy

    @property
    def vwap_sell(self):
        """
        Vwap sell.
        """

        vwap_sell = None

        if AgentTrade.history:

            sell_trades = list(filter(lambda d: d['agent_side'] == 2,
                                              AgentTrade.history))

            if sell_trades:

                vwap_sell = sum(trade['execution_price'] *
                                trade['executed_volume']
                                for trade in sell_trades) / sum(
                                trade['executed_volume']
                                for trade in sell_trades) * 1e-8
                vwap_sell = round(vwap_sell, 4)

        return vwap_sell

    @property
    def vwap_score(self):
        """
        Current vwap score of agent. Notably, vwap scores can only be
        computed if there is trade volume in the respective episode hence
        it is less suitable for very short episodes.
        :return
        """
        vwap_score = None

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

            if vwap_buy or vwap_sell:

                realized_buy_trades = list(
                    filter(lambda d: d['agent_side'] == 1, realized_trades))

                realized_sell_trades = list(
                    filter(lambda d: d['agent_side'] == 2, realized_trades))

                qt_bought = sum(trade['executed_volume']
                                for trade in realized_buy_trades)
                qt_sold = sum(trade['executed_volume']
                              for trade in realized_sell_trades)

                vwap_score = (qt_bought * (vwap_symbol - vwap_buy) +
                              qt_sold * (vwap_sell - vwap_symbol))

        return vwap_score

    @property
    def pnl_realized(self):
        """
        Current realized PnL of agent.

        Note: The pln_realized variable can be zero even after a lot of trading
        activity took place when the vwap_buy and vwap_sell are equal.
        """
        pnl_realized = 0

        realized_quantity = self.realized_quantity
        vwap_buy = self.vwap_buy
        vwap_sell = self.vwap_sell

        if realized_quantity:

            if vwap_buy and vwap_sell:

                # net long
                if realized_quantity > 0:

                    pnl_realized = realized_quantity * (vwap_sell - vwap_buy)

                # net short
                elif realized_quantity < 0:

                    pnl_realized = realized_quantity * (vwap_buy - vwap_sell)

        return pnl_realized

    @property
    def pnl_unrealized(self):
        """
        Unrealized PnL.
        """
        unrealized_trades = self.get_unrealized_trades
        unrealized_quantity = self.unrealized_quantity

        pnl_unreal = 0

        if unrealized_trades:

            # average share price of unrealized trades
            asp = sum(trade['execution_price'] * trade['executed_volume']
                                 for trade in unrealized_trades) / sum(
                                 trade['executed_volume']
                                 for trade in unrealized_trades)

            # net long position -> sell for current best-bid
            if unrealized_quantity > 0:

                immediate_price = Market.instances['ID'].best_bid

            # net short position -> buy back for current best ask
            elif unrealized_quantity < 0:

                immediate_price = Market.instances['ID'].best_ask

            if unrealized_quantity != 0:

                pnl_unreal = (unrealized_quantity
                              * (immediate_price - asp)
                              * 1e-8 * 1e-4)

        return round(pnl_unreal, 2)

    def latest_trade_is(self,
                        number_of_latest_trades,
                        scaling_factor=-1000):
        """
        Implementation shortfall of the latest trade. This function can e.g. be
        called in Reward. Note: this way, the is does not account for the
        volume of the trade.
        :param number_of_latest_trades
            int, number of latest trades for which the weighted is should be
            computed.
        """
        last_is = 0
        realized_trades = AgentTrade.history

        if realized_trades:
            # Select the respective trades.
            latest_trades = realized_trades[-number_of_latest_trades:]
            sum_quantity = 0
            sum_weighted_is = 0
            for trade in latest_trades:
                is_side = 1 if trade['agent_side'] == 1 else -1
                execution_price = trade['execution_price']
                arrival_price = trade['arrival_price']
                quantity = trade['executed_volume']
                sum_quantity += quantity
                # see Velu p. 337
                trade_is = is_side*(
                        execution_price-arrival_price)/arrival_price
                weighted_is = trade_is * quantity
                sum_weighted_is += weighted_is
            # If positive, divide by sum
            if sum_weighted_is and sum_quantity:
                last_is = sum_weighted_is / sum_quantity

        # Reward is scaled with -100.
        return round(last_is*scaling_factor, 4)

    def overall_is(self, scaling_factor=-1000):
        """
        Volume weighted implementation shortfall of all filled trades.
        Note: If there are both buy and sell orders, overall implementation
        may not be very meaningful.
        """
        realized_trades = AgentTrade.history
        volume_sum = 0
        weighted_trade_is_sum = 0
        for trade in realized_trades:
            is_side = 1 if trade['agent_side'] == 1 else -1
            execution_price = trade['execution_price']
            arrival_price = trade['arrival_price']
            volume = trade['executed_volume']
            volume_sum += volume
            trade_is = is_side * (execution_price -
                                           arrival_price) / arrival_price
            weighted_trade_is_sum += trade_is*volume

        overall_is = weighted_trade_is_sum / volume_sum

        return round(overall_is*scaling_factor, 4)

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

    def __str__(self, time_presentation=None):
        """
        String representation.
        """
        # get current timestamp from market
        timestamp = Market.instances['ID'].timestamp
        dt_timestamp = Market.instances['ID'].timestamp_datetime

        if not time_presentation:

            # string representation
            string = f"""
            ----------------------------
            unixtime:      {timestamp}
            datetime:       {dt_timestamp}
            ---
            exposure:       {self.exposure}
            pnl_realized:   {self.pnl_realized}
            pnl_unrealized: {self.pnl_unrealized}
            ----------------------------
            """

        else:
            if time_presentation == 'unix':
                time = timestamp
            elif time_presentation == 'datetime':
                time = dt_timestamp

            # string representation
            string = f"""
            ----------------------------
            time:      {time}
            ---
            exposure:       {self.exposure}
            pnl_realized:   {self.pnl_realized}
            pnl_unrealized: {self.pnl_unrealized}
            ----------------------------
            """

        return textwrap.dedent(string)
