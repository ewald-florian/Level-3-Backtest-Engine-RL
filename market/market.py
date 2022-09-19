#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import pandas as pd

from reconstruction.reconstruction import Reconstruction
from agent.agent_trade import AgentTrade
from market.market_trade import MarketTrade
from agent.agent_order import OrderManagementSystem as OMS


# TODO: / Debug match_n() method (auch mit impact)
# TODO: If I remove liquidity from simulation state, I also have to do this
#  in the observation space (to have consistency)


class Market(Reconstruction):

    # store several instances in dict with market_id as key
    instances = dict()

    """
    # TODO: Write Market class docstring.
    """

    def __init__(self,
                 market_id: str = "ID",
                 l3_levels: int = 10,
                 l2_levels: int = 10,
                 report_state_timestamps: bool = True,
                 match_agent_against_exec_summary: bool = True,
                 agent_latency: int = 0,
                 track_index: bool = False):

        super().__init__(track_timestamp=True,
                         track_index=track_index)
        """
        Instantiate Market.
        
        :param market_id:
            str, ...
        :param l3_levels
            int, number of price levels presented in state_l3
        :param l2_levels
            int, number of price levels presented in state_l2
        :param report_state_timestamps
            Bool, if True, store timestamp in state_l1, state_l2, state_l3
        :param agent_latency
            int, latency of agent messages is nanoseconds
        :param: match_agent_against_exec_summary
            bool, if True, agent orders will be matched against exec summaries.
        :param track_index
            bool, if True, the internal state index will be tracked
        ...
        """
        # static attributes from arguments
        self.market_id = market_id
        self.l3_levels = l3_levels
        self.l2_levels = l2_levels
        self.report_state_timestamps = report_state_timestamps
        self.match_agent_against_exe_sum = match_agent_against_exec_summary
        self.agent_latency = agent_latency  # 8790127
        # dynamic attributes

        # update class instance
        self.__class__.instances.update({market_id: self})

    # properties . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @property
    def state_l3(self):
        """
        Level 3 limit order book representation (internal _state). state_l3 has
        a similar structure than the internal state:

        {Side: {Price_Level: [{'timestamp: X , 'quantity': X} , {...}],
        Price_Level: [{...}, {...}]...

        The difference is that for each order, state_l3 only contains timestamp
        and quantity (and price as key) while in the internal state, the orders
        contain template_id, meg_seq_num, side, price, quantity, timestamp.
        Hence most infos in the internal state are redundant or not relevant.

        If self.l3_level is defined, the output will contain
        only a limited number of price levels.

        :return: state_l3_output
            dict, state_l3 with only quantity and timestamp per order
        """

        # TODO: right now, it has the same speed than the old for-loop...

        state_l3 = {}

        # bid side (1), highest prices first
        bid_keys = sorted(list(self._state[1].keys()), reverse=True)

        if self.l3_levels:
            bid_keys = bid_keys[:self.l3_levels]

        bid_side_dict = self._state[1]

        # Note: this operation does not build reference to _state,
        # no copy/deepcopy necessary
        adj_bid_side_dict = {
            n: [{'quantity': d['quantity'], 'timestamp': d['timestamp']}
                for d in bid_side_dict[n]] for n in bid_keys}

        # ask side (2), lowest prices first
        ask_keys = sorted(list(self._state[2].keys()), reverse=False)

        if self.l3_levels:
            ask_keys = ask_keys[:self.l3_levels]

        ask_side_dict = self._state[2]

        adj_ask_side_dict = {
            n: [{'quantity': d['quantity'], 'timestamp': d['timestamp']}
                for d in ask_side_dict[n]] for n in ask_keys}

        # add timestamp
        if self.report_state_timestamps:
            state_l3[0] = self.timestamp

        # combine to state_l3
        state_l3[1] = adj_bid_side_dict
        state_l3[2] = adj_ask_side_dict

        return state_l3

    @property
    def state_l2(self):
        """
        Level 2 limit order book representation with aggregated quantities per
        price level, structure. If self.l2_level is defined, the output will
        contain only a limited number of price levels:

        {1(Bid-Size): {Price 1: Aggregated Quantity, Price 2: Aggregated
        Quantity, ... }, 2(Ask Side): {...}}

        :return state_l2
            dict, level_l2 representation of internal state
        """
        keys = []
        state_l2 = {}
        # if activated, save current timestamp to state_l2[0]
        if self.report_state_timestamps:
            state_l2[0] = self.timestamp

        state_l2 = {}
        for side in [1, 2]:
            if side == 1:
                keys = sorted(self._state[side].keys(), reverse=True)
            elif side == 2:
                keys = sorted(self._state[side].keys(), reverse=False)

            if self.l2_levels:
                keys = keys[:self.l2_levels]

            side_dict = self._state[side]
            # aggregate quantities on price levels
            agg_side_dict = {n: sum([(d['quantity']) for d in side_dict[n]])
                             for n in keys}

            state_l2[side] = agg_side_dict

        return state_l2

    @property
    def state_l1(self):  # transform l3 into l1
        """
        Level 1 limit order book representation. Structure:

        {0: Timestamp, 1: {Best Bid Price: Best Bid Quantity},
        2: {Best Ask Price: Best Ask Quantity}}

        :return state_l1
            dict, Level 1 representation
        """
        state_l1 = {}

        # if activated, save current timestamp to state_l1[0]
        if self.report_state_timestamps:
            state_l1[0] = self.timestamp

        best_bid = max(self._state[1].keys())
        best_ask = min(self._state[2].keys())

        # TODO: sum list comprehension
        bid_quantity = 0
        for order in self._state[1][best_bid]:
            bid_quantity += order["quantity"]

        state_l1[1] = {best_bid: bid_quantity}

        ask_quantity = 0
        for order in self._state[2][best_ask]:
            ask_quantity += order["quantity"]

        state_l1[2] = {best_ask: ask_quantity}

        return state_l1

    @property
    def orderpool(self):
        """
        List with all order-messages which are stored in the internal state.
        :return: orderpool
            list, contains all orders
        """
        orderpool = []
        for side in [1, 2]:
            for price_level in self._state[side].keys():
                for order in self._state[side][price_level]:
                    orderpool.append(order)
        return orderpool

    @property
    def ticksize(self):
        """
        Ticksize.
        :return: ticksize
            int,
        """
        prices = list(self._state[1].keys()) + list(self._state[2].keys())
        ticksize = math.gcd(*prices)
        return ticksize

    #TODO: remove unnecessary properties but check if I used them somewhere! (e.g. I think I used best_ask somewhere)
    @property
    def best_bid(self):
        """
        Best bid price of current state.
        :return: best bid
            int, best bid price of current state.
        """
        return max(self._state[1].keys())

    @property
    def best_ask(self):
        """
        Best ask price of current state.
        :return: best ask
            int, best ask price of current state.
        """
        return min(self._state[2].keys())

    @property
    def midpoint(self):  # relevant only in exchange-based setting
        """
        Current index ('msg_seq_num'). Relevant only in exchange-based setting.
        :return midpoint
            int, midpoint of current state
        """
        best_bid = max(self._state[1].keys())
        best_ask = min(self._state[2].keys())
        midpoint = ((best_bid + best_ask) / 2)

        return midpoint

    @property
    def spread(self):
        """
        Spread of current state.
        :return: spread
            int, spread of current state.
        """
        best_bid = max(self._state[1].keys())
        best_ask = min(self._state[2].keys())
        spread = best_ask - best_bid

        return spread

    @property
    def relative_spread(self):
        """
        Relative Spread of current state.
        :return: relative spread
        """
        spread = self.spread()
        midpoint = self.midpoint()
        return spread / midpoint

    @property
    def index(self):
        """
        Current index ('msg_seq_num'). Relevant only in
        exchange-based setting.
        """
        if self.track_index:
            return self._state_index

    @property
    def timestamp(self):  # relevant only in agent-based setting
        """
        Current timestamp ('timestamp') in UTCT. Relevant only
        in agent-based setting.
        :return: timestamp
            int, unix time
        """
        return self._state_timestamp

    @property
    def timestamp_datetime(self):
        """
        Timestamp converted to datetime.
        :return: timestamp,
            datetime
        """
        utct_timestamp = int(self._state_timestamp)
        datetime_timestamp = pd.to_datetime(utct_timestamp, unit='ns')
        return datetime_timestamp

    @staticmethod
    def _store_market_trades(message_packet):
        """
        Trades are stored in MarketTrade.history.

        Execution summary messages do not need to be processed to maintain the
        internal state since the respective orders never reached the orderbook.
        They can bes used to produce a trade history which is useful to obtain
        market stats such as market VWAP. 13202 messages give a summary of an
        entire matching event while 13104/05 messages can be used to retrieve
        the individual trades.
        :param message_packet:
            list, contains messages
        """
        exec_sum_message = list(filter(lambda d: d['template_id'] == 13202,
                                       message_packet.copy()))[0]
        time = exec_sum_message['exec_id']
        aggressor_side = exec_sum_message['side']

        for message in message_packet:
            if message['template_id'] in [13104, 13105]:
                # additional info from 13202
                message['execution_time'] = time
                message['aggressor_side'] = aggressor_side
                # store to MarketTrade.history
                MarketTrade(market_trade=message)

    #  agent order simulation WITH MARKET IMPACT . . . . . . . . . . . . . . .

    # TODO: Überarbeiten damit es mit MarketInterface kompatibel ist!
    def update_with_agent_message_impact(self, message):
        """
        This method is the entry point of MarketInterface to submit and cancel
        agent orders with impact.

        Updates are based on agent-based message, a simple decision to either
        submit (99999) or cancel (66666) a message that will potentially lead
        to a crossed limit order book pre-match.
        Consequently, a matching step must follow. The idea is to "first add
        all liquidity and then take away matched
        liquidity".

        Submission orders have to be indicated with template_id = 99999.
        Cancellation messages have to be indicated with template_id = 66666.

        Orders submitted via update_with_agent_message() are stored in the
        actual internal state and can affect the market. When they are
        executed, they take liquidity of historic orders (market impact).

        :param message:
            dict, agent message
        :return trade_list:
            list, dicts with executed trades
        """
        # append message to OMS.order_list
        OMS(message)
        # update internal limit order book state ...

        message['timestamp'] = self.timestamp + self.agent_latency

        # TODO: add key to indicate that indicate
        # order submit
        if message["template_id"] == 99999:
            self._order_submit_impact(message)
        # order cancel
        elif message["template_id"] == 66666:
            self._order_cancel_impact(message)

        # TODO: match und match_new checken ob gleiche results
        # match internal limit order book state
        # trade_list = self.match() # Old...
        trade_list = self.match_new(state_to_match=self._state)  # New...
        # removes empty price levels from state
        self._clean_state()

        return trade_list

    def _order_submit_impact(self, message):
        """
        Process agent submission message.
        :param message
            dict, agent submission message
        """

        # ensure that 'msg_seq_num' field exists for compatibility reasons
        message["msg_seq_num"] = None

        # extract message information
        side = message["side"]
        price = message["price"]
        quantity = message["quantity"]

        # if price level not existing, create price level
        if price not in self._state[side].keys():
            self._state[side][price] = []
        # add order
        # TODO: remove .get() use [price]?
        self._state[side].get(price, []).append(message)

    def _order_cancel_impact(self, message):
        """
        Process agent cancellation message. The order which is to be cancelled
        can be identified by the combination of price, side and timestamp.
        The cancellation message needs to have the attribute side, price and
        timestamp.
        :param message
            dict, agent cancellation message
        """
        # extract message information
        side = message["side"]
        price = message["price"]
        timestamp = message["timestamp"]

        try:

            for idx, order in enumerate(self._state[side][price]):
                # identify order by timestep and remove order
                if order["timestamp"] == timestamp:
                    self._state[side][price].remove(order)
                    # log cancellation:
                    print(
                        f'(INFO) Agent Order Cancelled: Side: {side} | Price: {price} | Timestamp: {timestamp}')

            for position, message in enumerate(self._state[side][price]):
                if message["timestamp"] == timestamp:
                    del self._state[side][price][
                        position]
                    # if no order left on price level, delete price level
                    if not self._state[side][price]:
                        del self._state[side][
                            price]
                    return True

        except Exception as error:
            print(error)

    def _order_amend(self):  # equivalent to 'cancel and resubmit'
        pass

    #  agent order simulation WITHOUT MARKET IMPACT . . . . . . . . . . . . .

    #TODO: better name for this method
    def update_simulation_with_exchange_message(self, message_packet):
        """
        This method is the entry point for Replay to Marekt to step the
        market environment.

        Update market messages and check if agent matching is possible.
        """

        # -- call the reconstruction method to process historical messages.

        # returns the parsed message_packet
        parsed_message_packet = self.update_with_exchange_message(
            message_packet)

        # -- process execution summary messages

        if list(filter(lambda d: d['template_id'] == 13202,
                       parsed_message_packet)):
            # store market trades
            self._store_market_trades(parsed_message_packet)

            # match agent against execution summary (if activated)
            if self.match_agent_against_exe_sum:
                self._match_agent_against_execution_summary(
                    parsed_message_packet)

        # -- check if agent orders can be matched against book
        # Note: changes which can lead to matching only come from order_add
        # (13101) or order_modify (13101)
        if list(filter(lambda d: d['template_id'] in [13100, 13101],
                       parsed_message_packet)):

            self._simulate_agent_order_matching()

    def update_simulation_with_agent_message(self, message):
        """
        This method is the entry point of MarketInterface to submit and cancel
        simulated agent orders without impact.

        Process simulated agent messages:

        Each incoming agent message gets the current state timestamp plus
        latency as message timestamp.

        Submission messages (template_id=99999) get appended to the OMS.
        Cancellation messages (template_id=66666) are executed by changing the
        template_id of the cancelled order message from 99999 to 33333. To
        cancel an order it has to be identifiable by the price, side,
        timestamp combination.

        Cancellation messages are also appended to the OMS to have a complete
        protocol.

        Messages submitted via simulated_update_with_agent_message() are
        stored and matched separately from the internal state and do not affect
        the market (no market impact).

        :param message:
            dict, agent message
        :return:
        """
        # set current statetime plus latency as agent message timestamp

        message['timestamp'] = self.timestamp + self.agent_latency
        message['msg_seq_num'] = None
        message['message_id'] = len(OMS.order_list)

        # Store the agent order to OMS
        if message['template_id'] == 99999:
            OMS(message)
            # test if agent order matching is possible
            self._simulate_agent_order_matching()

        elif message['template_id'] == 66666:
            # append cancellation message to OMS
            OMS(message)
            cancelled_order = list(filter(
                lambda d: d['message_id'] == message['order_message_id'],
                OMS.order_list))[0]
            # change template_id from 99999 to 33333 to mark as CANCELLED
            cancelled_order['template_id'] = 33333
        else:
            print('(WARNING) agent message template_id not valid.')

    def _simulate_agent_order_matching(self):
        """
        Method to match simulated agent orders with the internal state.
        The agent orders are stored as messages is OMS and matched
        with simulation_state, a light version copy of the internal market
        state which only contains relevant price levels (for higher efficiency)
        """
        trade_list = None
        # -- build simulation state

        simulation_state = self._build_simulation_state()

        # -- match simulation_state, receive trade list
        if simulation_state:
            trade_list = self.match_new(state_to_match=simulation_state)

        # -- update OMS
        if trade_list:
            self._process_executed_agent_orders(trade_list)

        # -- store executed orders to AgentTrade
        if trade_list:
            self._store_agent_trades(trade_list)

    def _build_simulation_state(self):
        """
        Build the simulation state.
        :return:
        """
        simulation_state = {}
        # check if active (99999) agent messages with active timestamp exist
        if list(filter(lambda d: d['template_id'] == 99999 and
                                 d['timestamp'] <= self.timestamp,
                       OMS.order_list)):

            # dicts to create simulation_state
            bid_side_dict = {}
            ask_side_dict = {}
            # lists for values to calculate thresholds
            # (lists needed if agent messages exist only on one side)
            bid_threshold_values = []
            ask_threshold_values = []

            # -- define thresholds for relevant state
            # (i.e. which price levels need to be present)

            # filter prices from agent messages
            buy_prices = []
            sell_prices = []

            for message in OMS.order_list:
                # filter for order submissions (exclude cancellations):
                # Note: account for LATENCY via timestamp
                if message['template_id'] == 99999 and message[
                                'timestamp'] <= self.timestamp:
                    if message['side'] == 1:
                        buy_prices.append(message['price'])
                    if message['side'] == 2:
                        sell_prices.append(message['price'])

            # compute min/max buy/sell prices
            if len(buy_prices) > 0:
                max_buy_order_price = max(buy_prices)
                bid_threshold_values.append(max_buy_order_price)
                min_buy_order_price = min(buy_prices)
                ask_threshold_values.append(min_buy_order_price)
            if len(sell_prices) > 0:
                max_sell_order_price = max(sell_prices)
                ask_threshold_values.append(max_sell_order_price)
                min_sell_order_price = min(sell_prices)
                bid_threshold_values.append(min_sell_order_price)

            # compute thresholds:
            ask_threshold = max(ask_threshold_values)
            bid_threshold = min(bid_threshold_values)

            # Bid-side-keys
            bid_keys = list(self._state[1].keys())
            bid_keys_relevant = list(i for i in bid_keys if i >= bid_threshold)
            # Ask-side-keys
            ask_keys = list(self._state[2].keys())
            ask_keys_relevant = list(i for i in ask_keys if i <= ask_threshold)

            # -- construct simulation_state from self._state

            # store relevant bid levels to bid_side_dict
            # Note: According to my tests, copy is sufficient to not
            # affect internal state
            for key in bid_keys_relevant:
                if key in self._state[1]:
                    bid_side_dict[key] = self._state[1][key].copy()

            # store relevant ask levels to ask_side_dict
            for key in ask_keys_relevant:
                if key in self._state[2]:
                    ask_side_dict[key] = self._state[2][key].copy()

            # store relevant levels to simulation_state
            simulation_state[1] = bid_side_dict
            simulation_state[2] = ask_side_dict

            # TODO: Remove liquidity used by the agent earlier in this episode (make as optional setting) BEFORE agent masseges are appendet!
            # TODO: If I remove liquidity, I also have to do this in the observation space (to have consistency)!!!
            # -- add agent messages to simulation_state:

            # Note1: messages can just be appended, they will be executed according to their priority time
            # Note2: I copy the messages instead of giving a reference to the original message, this means that
            # hence, massages in the OMS will not directly be affected by matching.
            # Note3: account for LATENCY by selecting only messages with valid timestamp

            for message in OMS.order_list:

                if message['template_id'] == 99999 and message[
                                        'timestamp'] <= self.timestamp:
                    # buy-order
                    if message['side'] == 1:
                        price = message['price']
                        # add message if price level exists
                        if price in simulation_state[1]:
                            simulation_state[1][price].append(message.copy())
                        # create new price level for message
                        else:
                            simulation_state[1][price] = [message.copy()]

                    # sell-order
                    if message['side'] == 2:
                        price = message['price']
                        if price in simulation_state[2]:
                            simulation_state[2][price].append(message.copy())
                        else:
                            simulation_state[2][price] = [message.copy()]

        return simulation_state

    # debugged and tested (15.19.2022)
    @staticmethod
    def _process_executed_agent_orders(trade_list):
        """
        Helper method to process executed agent messages in the OMS.
        If an agent order is fully
        executed, its template_id gets updated to 11111. If an agent order
        is partially executed, the execution volume is deducted from the
        quantity and the order remains active (template_id = 99999). To keep
        track of partial executions, their volume is stored in 'partial
        executions'.

        :param trade_list
            list, contains execution summaries from match()
        """
        # iterate over trades
        for trade in trade_list:
            # message_id is unique identifier of message
            if 'message_id' in trade.keys():
                message_id = trade['message_id']
                executed_volume = trade['quantity']
                # TODO: actually relevant for Trade, but could also be saved in order...
                execution_price = trade['price']
                execution_time = trade['timestamp']

                # -- filter out the affected agent messages by message-id
                # Note: executed_order is reference to mutable message object
                # -> message can be manipulated
                # TODO: use d instead of message
                message = next(
                    filter(lambda message: message['message_id'] == message_id,
                           OMS.order_list))

                # -- manipulate the agent message to process the execution

                # agent orders was fully executed (possibly after part.ex.)
                if executed_volume >= message['quantity']:
                    message['template_id'] = 11111  # -11111: fully executed
                    if 'partial_executions' in message.keys():
                        message['partial_executions'].append(
                            message['quantity'])

                # agent order was partially executed
                if executed_volume < message['quantity']:
                    # add partial execution indicator
                    if 'partial_executions' in message.keys():
                        message['partial_executions'].append(executed_volume)

                    else:
                        message['partial_executions'] = [
                            executed_volume]  # list
                        message['original_quantity'] = message['quantity']

                    # remove executed quantity
                    message['quantity'] = message['quantity'] - executed_volume

    @staticmethod
    def _store_agent_trades(trade_list):
        """
        Store agent trade summaries to AgentTrade.history list.
        :param trade_list,
            list, contains execution summaries from match()
        """
        for trade in trade_list:
            agent_trade = {'trade_id': len(AgentTrade.history),
                           'execution_time': trade['timestamp'],
                           'executed_volume': trade['quantity'],
                           'execution_price': trade['price'],
                           'aggressor_side': trade['aggressor_side'],
                           'message_id': trade['message_id'],
                           "agent_side": trade["agent_side"]}

            # TODO: better to just call the class or make composition
            #  (e.g. Market.agent_trade = AgentTrade.history)
            AgentTrade(agent_trade)

    def _match_agent_against_execution_summary(self,
                                               message_packet):
        """
        Test whether agent orders from the agent_order_list could be matched
        against market- or marketable limit orders which never reach the
        orderbook. Compare agent orders to fully executed (template_id 13104)
        and partially executed (template_id 13105) orders and check if
        simulated agent orders have a higher price-time priority.

        If agent orders are executable, the _process_agent_execution() and the
        _store_agent_trade() methods are called to simulate the execution.

        _match_agent_against_execution_summary() method should be called if
         template_id of the most recent message is 13202 (Execution Summary),
         for example in the method update_with_exchange_message()

        Note: It is assumed that the originally executed orders (13204/05) are
        still executed to not deviate too far from the historical orderbook.

        :param message_packet,
            list, contains dictcs with messages
        """
        trade_list = []

        # check if active (99999) agent messages exist
        if list(filter(lambda d: d['template_id'] == 99999 and
                                 d['timestamp'] <= self.timestamp,
                       OMS.order_list)):

            # -- filter for the execution summary message to get execution info
            exec_sum_message = list(filter(lambda d: d['template_id'] == 13202,
                                           message_packet.copy()))[0]

            aggressor_side = exec_sum_message['side']  # "AggressorSide"
            # worst price of this match (aggressor viewpoint).
            last_price = exec_sum_message['price']  # LastPx
            aggressor_timestamp = exec_sum_message[
                'timestamp']  # AggressorTime

            # -- get historically executed orders from message_packet
            executed_orders = list(
                filter(lambda d: d['template_id'] in [13104, 13105],
                       message_packet.copy()))

            # -- get potentially executable agent orders from OMS
            # (side, ACTIVE (99999), active timestamp)
            active_agent_orders = list(
                filter(lambda d: d['template_id'] == 99999 and
                                 d['timestamp'] <= self.timestamp,
                       OMS.order_list.copy()))

            complementary_agent_orders = list(
                filter(lambda d: d['side'] != aggressor_side,
                       active_agent_orders))

            # agent-ask <= aggressor bid:
            if aggressor_side == 1:
                executable_agent_orders = list(filter(lambda d: d[
                        'price'] <= last_price, complementary_agent_orders))
            # agent-bid >= aggressor ask:
            elif aggressor_side == 2:
                executable_agent_orders = list(filter(lambda d: d[
                        'price'] >= last_price, complementary_agent_orders))

            # skip if no executable agent orders exist
            if len(executable_agent_orders) > 0:

                # sort executable_agent_orders by price and timestamp
                if aggressor_side == 2:
                    # sort by highest price and smallest timestamp
                    executable_agent_orders = sorted(executable_agent_orders,
                                key=lambda d: (-d['price'], d['timestamp']))
                elif aggressor_side == 1:
                    # sort by lowest price and smallest timestamp
                    executable_agent_orders = sorted(executable_agent_orders,
                                key=lambda d: (d['price'], d['timestamp']))

                # -- test if agent orders would have had a higher priority
                # than historically executed orders

                for agent_order in executable_agent_orders:

                    message_id = agent_order["message_id"]

                    for executed_order in list(executed_orders):

                        # set execution flag to false
                        agent_execution_possible = False
                        # Note: if the agent order has priority, the agent
                        # order price is relevant, could potentially be worse
                        # than the original execution price
                        execution_price = executed_order['price']
                        execution_quantity = min(executed_order['quantity'],
                                                 agent_order['quantity'])

                        if aggressor_side == 1 and agent_order['price'] < \
                                executed_order['price']:
                            agent_execution_possible = True

                        elif aggressor_side == 2 and agent_order['price'] > \
                                executed_order['price']:
                            agent_execution_possible = True

                        # smaller timestamp has higher priority
                        elif agent_order['price'] == executed_order[
                            'price'] and agent_order['timestamp'] < \
                                executed_order[
                                    'timestamp']:
                            agent_execution_possible = True

                        if agent_execution_possible:
                            match_execution_summary = {
                                "aggressor_side": aggressor_side,
                                "price": execution_price,
                                "timestamp": aggressor_timestamp,
                                "quantity": execution_quantity,
                                "message_id": message_id,
                                "agent_side": agent_order['side']}

                            trade_list.append(match_execution_summary)
                            # remove executed_order from executed_orders
                            executed_orders.remove(executed_order)

            # process agent executions
            if trade_list:
                self._process_executed_agent_orders(trade_list)
                self._store_agent_trades(trade_list)

    # debugged (15.09.22), further testing required (impact orders)
    @staticmethod
    def match_new(state_to_match):
        """
        Idea: use the same matching method for simulated matching without
        market impact and "real" matching with market impact by passing the
        respective state (internal_state or simulation_state) as input
        argument.

        The simulated state is entirely copied and does not include references
        to the internal state or the OMS. Hence, manipulating the simulation
        state does neither affect the internal state nor the OMS. Instead,
        agent messages are executed separately in ...

        :param state_to_match,
            dict, state which should be matched (either simulation_state or
            self._state)
        """
        # check if state_to_match has two order book sides

        if state_to_match[1] and state_to_match[2]:

            trade_list = []

            # loop until best price levels cannot be matched
            while True:

                # break if not both sides are filled with orders
                if not state_to_match[1] or not state_to_match[2]:
                    break

                max_buy = max(state_to_match[1].keys())  # best bid
                min_sell = min(state_to_match[2].keys())  # best ask

                # break if order book not crossed
                if not max_buy >= min_sell:
                    break

                # if empty, remove the current best price_level and continue
                if not (price_level := state_to_match[1][
                    max_buy]): #:=allows variable assignment inside expression
                    del state_to_match[1][max_buy];
                    continue

                if not (price_level := state_to_match[2][min_sell]):
                    del state_to_match[2][min_sell];
                    continue

                # sort by priority time and select first order (smallest
                # priority timestamp), necessary for simulation!
                order_buy = sorted(state_to_match[1][max_buy],
                                   key=lambda d: d['timestamp'])[0]
                order_sell = sorted(state_to_match[2][min_sell],
                                    key=lambda d: d['timestamp'])[0]

                # aggressor order has later timestamp
                order_standing, order_aggressor = sorted(
                    [order_buy, order_sell], key=lambda x: x["timestamp"])

                # execution price is always the price of the standing order
                execution_price = order_standing["price"]
                aggressor_side = order_aggressor["side"]
                aggressor_timestamp = order_aggressor["timestamp"]

                if order_buy['quantity'] < order_sell['quantity']:
                    # remove qt from partially executed sell order
                    order_sell["quantity"] -= order_buy["quantity"]
                    # save executed quantity for trade report
                    execution_quantity = order_buy["quantity"]
                    # remove the fully executed buy order from state
                    state_to_match[1][max_buy].remove(order_buy)

                elif order_buy['quantity'] > order_sell['quantity']:
                    # remove qt from partially executed buy order
                    order_buy["quantity"] -= order_sell["quantity"]
                    # store to trade-list
                    execution_quantity = order_sell["quantity"]
                    # remove executed sell order from simulation state
                    state_to_match[2][min_sell].remove(order_sell)

                else:
                    # both orders fully executed
                    execution_quantity = order_sell["quantity"]
                    # remove both fully executed orders from simulation state
                    state_to_match[1][max_buy].remove(order_buy)
                    state_to_match[2][min_sell].remove(order_sell)

                # append "execution summary" to trade list
                match_execution_summary = {"aggressor_side": aggressor_side,
                                           "price": execution_price,
                                           "timestamp": aggressor_timestamp,
                                           "quantity": execution_quantity,
                                           }

                # if agent-message was matched, add message_id
                if "message_id" in order_buy.keys():
                    match_execution_summary["message_id"] = order_buy[
                        "message_id"]
                    match_execution_summary["agent_side"] = 1
                elif "message_id" in order_sell.keys():
                    match_execution_summary["message_id"] = order_sell[
                        "message_id"]
                    match_execution_summary["agent_side"] = 2
                # TODO: Eigenausführungen verhindern?
                # Edge-Case, both sides are agent orders:
                elif "message_id" in order_buy.keys() and "message_id" in \
                        order_sell.keys():
                    match_execution_summary["message_id"] = order_sell[
                        ["message_id"],
                        order_buy["message_id"]]
                else:
                    pass

                trade_list.append(match_execution_summary)

            return trade_list

        else:
            print('(WARNING) State is not complete - no matching possible')

    @classmethod
    def reset_instances(cls):
        """
        Reset all market instances by clearing the instances dictionary.
        """
        # delete all elements in MarketState.instances (dictionary)
        cls.instances.clear()


