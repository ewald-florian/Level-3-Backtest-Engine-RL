#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# v2: - Add agent order simulation (agent_order_list etc.)
# - self.agent_order_list

# Track Modifications:
# - fixed bug from matching (false execution price in trade list)
# - Added class instance
# - Added logging (print weil logging verzögert)
# - added _update_internal_timestamp(), _update_index()
# - fixed bug in level_l1 -> keys were not sorted which led to false best bid ask prices
# - added best_bid, best_ask, spread, timestamp_datetime properties
# - _order_modify(): change message_keep["template_id"] = 13101 (instead of 13100) because
#   13100 led to a snapshot-state discrepancy...
# - I removed, state_dict, order_pool, midpoints
# - I changed self.msg_counter to count every message instead of only order add messages
# - I removed cancel_training_messages(), sample_random_message()
# - I removed self.state_updated (MarketState is stateful object, history can be stored in Context)
# - I removed self.messages
# - Removed state_l2_array
# - added self.l3_levels and self.l2_levels args to save computational power if limited levels are sufficient
# - added self.report_state_timestamps flag (arg) to store current timestamp in state_l1, state_l2, state_l3 representations
# - added self.agent_message_list (mostly for debugging)
# - added loop to remove 'time-in' from internal_state in validate_state()
# - added orderpool
# - implemented _execution_summary() to store trade execution info to self.trade_execution_history list
# - added trade_history_array(), trade_history_df(), market_vwap() based on trade_execution_history list



import json
import copy
import logging # DEBUG, WARNING, INFO, CRITICAL, ERROR
logging.getLogger().setLevel(logging.INFO)

import pandas as pd
import numpy as np

from market.parser import MessagePacketParser
from market.parser import SnapshotParser

#TODO:
# - Test / Debug match() method
# - Brauche ich message_counter?
# - update_internal_timestamp und update_internal:index sind sehr ineffizient, optimieren...
# - class attribute better as dictionary / list? as in L2...
# - Idee wie execution summaries und simulated orders zusammengebrcuht werden können finden...
# - make update index optional (this could save a little bit of comp. effort)
# - TODO: If I remove liquidity from simulation state, I also have to do this in the observation space (to have consistency)!!!
# - TODO Execution Summarx: last Px is the "worst" execution price, not the only execition price, to be more
# - accurate, I have to get the trades from the 13104/13105 messages instead of 13202!

# --------------- MARKET STATE WITH CLASS ATTRIBUTE EXTENSION __________________________
class MarketStateAttribute:

    #TODO: test if parallelization works...Idea: use current ns timestamp as market_id

    # class attribute
    instance = None

    def __init__(self,
                 market_id: str,
                 l3_levels:int=None,
                 l2_levels:int=None,
                 report_state_timestamps:bool=False):
        """
        State is implemented as a stateful object that reflects the current
        limit order book snapshot in full depth and full detail.

        The state consists of a dictionary with two sides, which have the integer keys 1 (=Buy) and 2 (=Sell). Each of
        these sides contains a dictionary with price levels. The keys for the price levels are the prices as integers.
        Since the structure of a dictionary does not require the keys to be in order, the price levels in the dict may
        not be in order. Each price level contains a list of orders, which are in price-time-order. The orders are
        represented as dicts with the following values: "template_id", "msg_seq_num", "side", "price", "quantity"
        and "timestamp". All orders must contain the same six values and all of these values must be integers:

        Possible values:    "template_id":  13100 (order_add), 13101 (order_modify), 13102 (order_delete),
                                            13103 (order_mass_delete), 13104 (execution_full), 13105 (execution_partial),
                                            13106 (order_modify_same_priority), 13202 (execution_summary),
                                            99999 (order_submit agent-side), 66666 (order_cancel agent-side)
                            "msg_seq_num":  any integer, but must be in ascending order
                            "side": 1 (buy side) or 2 (sell side)
                            "price":    any integer, that is a valid pirce for the security. Note that the price is always
                                        represented as an integer, e.g. 123.45 would be 12345.
                            "quantity": any integer, that is a valid quantity for the security
                            "timestamp":    an integer that represents a unixtimestamp

        :param market_id:
            str, ...
        :param l3_levels
            int, number of price levels presented in state_l3
        :param l2_levels
            int, number of price levels presented in state_l2
        :param report_state_timestamps
            Bool, if True, store timestamp in state_l1, state_l2, state_l3
        :param
        ...
        """

        # static attributes from arguments
        self.market_id = market_id
        self.l3_levels = l3_levels
        self.l2_levels = l2_levels
        self.report_state_timestamps = report_state_timestamps

        # dynamic attributes
        self._state = None
        self.msg_counter = 0
        # NEW: list to store agent messages (for debugging purposes)
        self.agent_message_list = []
        # NEW list to store historical trade executions (for executions from messages)
        self.trade_execution_history = []
        # NEW: list to store submitted agent messages, used for simulated agent order executions
        self.agent_order_list = []
        # NEW: for development of Trade datastructure
        self.agent_trade_list = []
        # ensure that _state is set
        self._state_ready = False

        # use 'msg_seq_num' to validate exchange-based updates
        self._state_index = None
        # use 'timestamp' to validate agent-based update
        self._state_timestamp = None

        # -- update class attribute
        self.__class__.instance = self #._state

    # attributes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @property
    def state_l3(self):  # retrieve l3 (default)
        """
        Level 3 limit order book representation (internal _state). state_l3 has
        a similar structure than the internal state:

        {Side: {Price_Level: [{'timestamp: X , 'quantity': X} , {...}],
        Price_Level: [{...}, {...}]...

        The difference is that for each order, state_l3 only contains timestamp
        and quantity (and price as key) while in the internal state, the orders
        contain template_id, meg_seq_num, side, price, quantity, timestamp. Hence
        most infos in the internal state are redundant or not relevant.

        If self.l3_level is defined, the output will contain
        only a limited number of price levels.

        :return: state_l3_output
            dict, state_l3 with only quantity and timestamp per order
        """
        state_l3 = self._state
        state_l3_output = {}

        # if activated, save current timestamp to state_l3[0]
        if self.report_state_timestamps:
            state_l3_output[0] = self.timestamp

        # transform only l3_levels price levels to state_3 representation
        if self.l3_levels:

            for side in [1, 2]:

                side_dict = {}

                keys = list(state_l3[side].keys())

                if side == 1:
                    # Bid-Sort: reverse=True (descending)
                    keys.sort(key=None, reverse=True)
                elif side == 2:
                    # Ask-Sort: reverse=False (ascending)
                    keys.sort(key=None, reverse=False)

                # select respective price levels
                keys = keys[:self.l3_levels]

                for price_level in keys:
                    price_level_orders = []
                    for order in state_l3[side][price_level]:
                        price_level_orders.append({'timestamp': order['timestamp'], 'quantity': order['quantity']})
                    side_dict[price_level] = price_level_orders
                state_l3_output[side] = side_dict

        # transform entire state to state_l3 representation
        else:  # levels=None

            for side in [1, 2]:
                side_dict = {}
                # iterate over price_level instances per side
                for price_level in state_l3[side].keys():
                    price_level_orders = []
                    for order in state_l3[side][price_level]:
                        price_level_orders.append({'timestamp': order['timestamp'], 'quantity': order['quantity']})
                    side_dict[price_level] = price_level_orders
                state_l3_output[side] = side_dict

        return state_l3_output

    @property
    def state_l2(self):
        """
        Level 2 limit order book representation with aggregated quantities per
        price level, structure. If self.l2_level is defined, the output will contain
        only a limited number of price levels:

        {1(Bid-Size): {Price 1: Aggregated Quantity, Price 2: Aggregated Quantity, ... }, 2(Ask Side): {...}}

        :return state_l2
            dict, level_l2 representation of internal state
        """
        # TODO: bisschen komisch dass hier "state_l3" verwendet wird...
        state_l3 = self._state
        state_l2 = {}

        # if activated, save current timestamp to state_l2[0]
        if self.report_state_timestamps:
            state_l2[0] = self.timestamp

        # limited price levels
        if self.l2_levels:

            for side in [1, 2]:
                side_dict = {}

                keys = list(state_l3[side].keys())

                if side == 1:
                    # Bid-Sort: reverse=True (descending)
                    keys.sort(key=None, reverse=True)
                elif side == 2:
                    # Ask-Sort: reverse=False (ascending)
                    keys.sort(key=None, reverse=False)

                # select respective price levels
                keys = keys[:self.l2_levels]

                for price_level in keys:
                    quantity = 0
                    # iterate over orders in price level
                    for order in state_l3[side][price_level]:
                        # aggregate quantity
                        quantity += order["quantity"]

                    side_dict[price_level] = quantity

                state_l2[side] = side_dict

        # all price levels
        else: # self.l2_levels = None
            # iterate over side
            for side in [1, 2]:
                side_dict = {}
                # iterate over price_level instances per side
                for price_level in state_l3[side].keys():
                    quantity = 0

                    # iterate over order instances per side and price_level
                    for order in state_l3[side][price_level]:
                        quantity += order["quantity"]

                    side_dict[price_level] = quantity
                state_l2[side] = side_dict

        return state_l2

    @property
    def state_l1(self):  # transform l3 into l1
        """
        Level 1 limit order book representation. Structure:

        {1: {Best Bid Price: Best Bid Quantity}, 2: {Best Ask Price: Best Ask Quantity}}

        :return state_l1
            dict, Level 1 representation
        """
        state_l1 = {}

        # if activated, save current timestamp to state_l1[0]
        if self.report_state_timestamps:
            state_l1[0] = self.timestamp

        best_bid = max(self._state[1].keys())
        best_ask = min(self._state[2].keys())

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
        orderpool=[]
        for side in [1,2]:
            for price_level in self._state[side].keys():
                for order in self._state[side][price_level]:
                    orderpool.append(order)
        return orderpool

    # TODO: delete unnecessary properties (Market should not be used to compute features...)
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
    def index(self):  # relevant only in exchange-based setting
        """
        Current index ('msg_seq_num'). Relevant only in exchange-based setting.
        """
        return self._state_index

    @property
    def timestamp(self):  # relevant only in agent-based setting
        """
        Current timestamp ('timestamp') in UTCT. Relevant only in agent-based setting.
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

    # trade history . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @property
    def trade_history_array(self):
        """
        Provide trade history as np array.
        :return: trade_history_array
            np.array, trade history
        """
        # assert (len(self.trade_execution_history) > 0, f"no trade history")
        if len(self.trade_execution_history) > 0:
            # convert trade execution history to array
            trade_history_array = np.array(self.trade_execution_history)

            return trade_history_array

    @property
    def trade_history_df(self):
        """
        Provide trade hisory as dataframe.
        :return: trade_history_df
            pd.DataFrame, df with trade history
        """

        # assert (len(self.trade_execution_history) > 0, f"no trade history")
        if len(self.trade_execution_history) > 0:
            # convert trade execution history to array
            trade_history_array = np.array(self.trade_execution_history)
            # convert array to pandas dataframe
            cols = ["exec_id", "price", "quantity", "aggressor_side"]
            trade_history_df = pd.DataFrame(trade_history_array, columns=cols)

            return trade_history_df

    @property
    def market_vwap(self):
        """
        Market VWAP.
        :return: market_vwap
            int, market vwap of current episode
        """
        # check if trade history exists
        if len(self.trade_execution_history) > 0:

            trade_history_array = np.array(self.trade_execution_history)
            # vwap = sum(price_order_i*qt_order_i) / sum(qt_all)
            # TODO: update vwap computation if array format is changed
            market_vwap = int(np.sum(trade_history_array[:, 1] * \
                                trade_history_array[:, 2]) / \
                              np.sum(trade_history_array[:, 2]))

            return market_vwap

    # start/end snapshot . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @SnapshotParser.parse
    def initialize_state(self, snapshot):
        """
        Set current state to be the provided snapshot from which to start
        reconstruction.

        The original snapshot format is parsed by SnapshotParser (as decorator).

        The parsed snapshot contains all orders with side and price
        as keys. Orders contain the attributes 'template_id', 'msg_seq_num',
        'side', 'price', 'quantity', timestamp'. (Note: this could change when
        using other data or when the downloader is modified...).
        :param snapshot:
            dict, contains all orders with side and price as keys
        """
        # set initial internal state to be the start snapshot
        self._state = snapshot
        # set initial state index
        self._update_internal_state_index()
        # set initial timestamp
        self._update_internal_timestamp()

        # logging.info('State has been build from snapshot | StartTimestamp: {}'.format(self.timestamp_datetime))
        print(f'(INFO) State Build from Snapshot: Start Timestamp: {self.timestamp_datetime}')

    @SnapshotParser.parse
    def validate_state(self, snapshot):
        """
        Compare current _state to the specified snapshot in order to validate
        the reconstruction.

        Note: some parsed orders contain a "time-in" key for priority time which
        does not exist in the snapshot data, therefore the "time-in" elements
        need to be removed from _state in order to match the snapshot.

        :param snapshot:
            dict, contains all orders with side and price as keys
        :return: boolean, 1 if snapshots match, 0 otherwise
        :return snapshot
            dict, snapshot parsed with SnapshotParser decorator
        """
        # remove "time-it"
        # use deepcopy to not manipulate the actual internal state
        internal_state = copy.deepcopy(self._state)
        for side in [1, 2]:
            for key in internal_state[side].keys():
                for msg in internal_state[side][key]:
                    if 'time-in' in msg.keys():
                        del msg['time-in']

        return internal_state == snapshot

    # delete state . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def _clean_state(self):
        """
        Remove empty price levels from the current state.
        """
        for side in self._state.keys():
            # store copy of keys by using list(), because dict size changes in case of price level deletion
            side_key_list = list(self._state[side].keys())
            for price_level in side_key_list:
                if not self._state[side][price_level]:
                    # remove empty price level
                    del self._state[side][price_level]

    def _empty_state(self):
        """
        Set current state to be empty limit order book representation.
        """
        # buy-side = 1 and sell-side = 2
        self._state = {1: {}, 2: {}}  # there are no price levels

    # exchange-based post-match update . . . . . . . . . . . . . . . . . . . .

    @MessagePacketParser.parse
    def update_with_exchange_message(self, message_packet):
        """
        Updates are based on exchange-based message packet, a set of messages
        that describe in detail each change to be made to the limit order book
        post-match (`_order_add`, `_order_delete`, ...). The idea is to "add
        only unmatched liquidity at each time step".

        The messages are parsed to a format similar to:
        {'template_id': X, 'msg_seq_num': X, 'side': X, 'price': X, 'quantity': X,
        'timestamp': X, 'time-in': X}

        - time-in: TrdRegTSTimeIn

        Concrete format of parsed message depends on message type (template_id), see
        MessagePacketParser class for more information.

        :param message_packet:
            list, contains single messages as dicts, message types can be inferred from TemplateIDs
        :return trade_list:
            list, contains dict with execution messages
        """

        # assert that self._state exists
        assert self._state, \
            "(ERROR) note that an update cannot take place without self._state"

        # prepare trade_list
        trade_list = []

        # iterate over all messages in message_packet
        for message in message_packet:

            # assert that self._state_index is smaller than 'msg_seq_num'
            assert self._state_index < message["msg_seq_num"], \
                "(ERROR) note that an update requires 'msg_seq_num' to be larger than self._state_index"

            # count message
            self.msg_counter += 1

            # update internal limit order book state ...

            # order add
            if message["template_id"] == 13100:
                self._order_add(message)
            # order delete
            elif message["template_id"] == 13102:
                status = self._order_delete(message)
            # order delete mass
            elif message["template_id"] == 13103:
                self._order_delete_mass(message)
            # order modify
            elif message["template_id"] == 13101:
                self._order_modify(message)
            # order modify same priority
            elif message["template_id"] == 13106:
                self._order_modify_same_priority(message)
            # execution full
            elif message["template_id"] == 13104:
                trade_list.append(
                    self._execution_full(message)
                )
            # execution partial
            elif message["template_id"] == 13105:
                trade_list.append(
                    self._execution_partial(message)
                )
            # execution summary
            elif message["template_id"] == 13202:
                self._execution_summary(message)
                ################################
                print(10*'!', '13202')
                print('MESSAGE PACKET: ', message_packet)
                print('AGENT MESSAGE LIST: ', self.agent_message_list)
                self._match_agent_orders_against_execution_summary(message_packet)
                ################################
            else:
                pass

        # update timestamp
        self._update_internal_timestamp()
        # update state index
        self._update_internal_state_index()

        return trade_list

    def _update_internal_timestamp(self):
        """
        Update the internal timestamp which is based on the largest
        timestamp of the orders in the state.
        """
        # search state for highest timestep
        max_timestamp = 0
        for side in self._state.keys():
            for price_level in self._state[side].keys():
                for order in self._state[side][price_level]:
                    if (timestamp := order["timestamp"]) > max_timestamp:
                        max_timestamp = timestamp

        # set internal state_timestamp to be highest 'timestamp' included in snapshot
        self._state_timestamp = max_timestamp

    def _update_internal_state_index(self):
        """
        Update the internal state index which is based on the largest
        msg_seq_num of the orders in the state.
        """

        max_index = 0

        # search state for highest 'msg_seq_num'
        for side in self._state.keys():
            for price_level in self._state[side].keys():
                for order in self._state[side][price_level]:
                    # Agent Masseges have None as msg_seq_num...
                    if order["msg_seq_num"]:
                        if (index := order["msg_seq_num"]) > max_index:
                            max_index = index

        self._state_index = max_index

    # helper functions to update state from messages . . . . . . . . . . . . .

    def _order_add(self, message):  # 13100
        """
        Add order-add (13100) message to _state.
        :param message:
            dict, message
        """

        # extract message information
        side = message["side"]
        price_level = message["price"]

        # cast message side to string
        message["side"] = side

        # if price level not existing, create price level
        if price_level not in self._state[side].keys():
            self._state[side][price_level] = []
        # add order to limit order book
        self._state[side].get(price_level).append(message)

    def _order_delete(self, message):  # 13102
        """
        Delete order specified in order delete message (13102) from _state.
        The order to delete is identified by side, price and timestamp.
        :param message:
            dict, message
        :return: Bool
            True if order was deleted
        """
        # extract message information
        side = message["side"]
        price_level = message["price"]
        timestamp = message["timestamp"]

        # delete specified order in limit order book based on side, price, and timestamp
        for position, message in enumerate(self._state[side][price_level]):
            # search for the order with the specific timestamp
            if message["timestamp"] == timestamp:
                del self._state[side][price_level][position]
                # if no order left on price level, delete price level
                if not self._state[side][price_level]:
                    del self._state[side][price_level]
                return True
        # ...
        else:
            return False

    def _order_delete_mass(self):  # 13103
        """
        Execute order mass delete message (13103) by emptying the state.
        """
        self._empty_state()

    def _order_modify(self, message):  # 13101
        """
        Execute order modify message (13101). Basically, the previous order
        is deleted and the modified version of the order is added.
        For standard modification, the timestamp is updated.
        :param message:
            dict, message
        """
        # delete order based on message_prev
        message_prev = {key.replace("prev_", ""): message[key] for key in [
            "msg_seq_num", "side",  # shared
            "prev_price", "prev_quantity", "prev_timestamp",
        ]}
        # order delete template_id (not so important since the order will not be stored in state...)
        message_prev["template_id"] = 13102 # order delete

        self._order_delete(message_prev)

        # add order with message_keep (last queue position)
        message_keep = {key: message[key] for key in [
            "template_id", "msg_seq_num", "side",  # shared
            "price", "quantity", "timestamp",
        ]}

        message_keep["template_id"] = 13101 # order modify

        # _order_add works independently of template_id
        self._order_add(message_keep)

    def _order_modify_same_priority(self, message):  # 13106
        """
        Execute order modify same priority message (13106).
        Priority Timestamp of order is not affected.
        :param message:
            dict, message
        """
        # extract message information
        side = message["side"]
        price_level = message["prev_price"]
        timestamp = message["prev_timestamp"]
        # replace target order with message_keep (same queue position)
        message_keep = {key: message[key] for key in [
            "template_id", "msg_seq_num", "side",
            "price", "quantity", "timestamp"
        ]}

        for position, message in enumerate(self._state[side][price_level]):
            if message["timestamp"] == timestamp:
                self._state[side][price_level][position] = message_keep

    def _execution_full(self, message):  # 13104
        """
        Extract trade information from execution full message (13104)
        and delete respective order from state.
        :param message:
            dict, message
        :return: trade
            dict, trade information
        """
        # extract message information
        side = message["side"]
        price_level = message["price"]
        timestamp = message["timestamp"]
        quantity = message["quantity"]  # executed quantity

        # delete order if order is fully executed
        self._order_delete(message)

        # trade report
        trade = {
            "aggressor_side": side,
            "price": price_level,
            "timestamp": timestamp,
            "executed_quantity": quantity,
        }

        return trade

    def _execution_partial(self, message):  # 13105
        """
        Extract trade information from execution partial message (13105)
        and delete respective (partial) order from state, i.e. if order is
        not fully executed, quantity is reduced by the executed quantity.

        Note: It is sufficient to process only full (13105) and partial
        executions (13104) since the execution summary (13202) only describes
        the perspective of the incoming (aggressor) order which is immediately
        executed and therefore never reaches the orderbook and hence does not
        need to be processed upon execution in order to maintain the state.

        :param message:
            dict, message
        :return: trade
            dict, trade information
        """
        # extract message information
        side = message["side"]
        price_level = message["price"]
        timestamp = message["timestamp"]
        quantity = message["quantity"]  # executed quantity

        # iterate over specified price level
        for position, message in enumerate(self._state[side][price_level]):
            if message["timestamp"] == timestamp:
                self._state[side][price_level][position]["quantity"] -= quantity  # subtract executed quantity
                # if order has quantity 0, delete order
                if self._state[side][price_level][position]["quantity"] == 0:
                    self._order_delete(message)

        # generate trade
        trade = {
            "aggressor_side": side,
            "price": price_level,
            "timestamp": timestamp,
            "executed_quantity": quantity,
        }

        return trade

    # TODO: 1304/1305 als basis für Trade list nehmen, 13202 ist nur eine Zusammenfassung!
    def _execution_summary(self, message):  # 13202
        """
        Execution summary messages do not need to be processed to maintain the
        internal state since the respective orders never reached the orderbook.
        They can bes used to produce a trade history which is useful to obtain
        market stats such as market VWAP.
        :param message:
            dict, execution_summary message
        """
        #TODO: Check if agent_orders could be matched against the execution-summary order
        # 1) Check if orders in self.agent_order_list could potentially be matched
        # 2) Check if these agent orders would have had a higher priority time than the
        # actually matched orders (tricky because they are already gone...)

        # retrieve trade infos
        exec_id = message['exec_id']
        #TODO: last Px is the "worst" execution price, not the only execition price, to be more
        # accurate, I have to get the trades from the 13104/13105 messages!
        price = message['price']
        quantity = message['quantity']
        aggressor_side = message['side']
        # store to array
        exec_array = np.array([exec_id, price, quantity, aggressor_side])
        # append array to trade_execution_history list
        self.trade_execution_history.append(exec_array)

    # agent-based pre-match update . . . . . . . . . . . . . . . . . . . . . .

    #TODO: This function could be called in MarketInterface to submit and cancel orders...from
    # - but how can the orders be tracked in the interface?
    # - :return some unique order attributes which allows it to identify orders in L3-state (->timestamp)
    # - then, market interface can filter l3 state for those orders and check whats going on...
    # - Later: create separate "L3_simulation_state" to not manipulate the historical market...from
    # - experiment with sustainably take liquidity from this simulation state...

    #  agent order simulation WITH MARKET IMPACT . . . . . . . . . . . . . . . . . . . . . .

    def update_with_agent_message(self, message):
        """
        Updates are based on agent-based message, a simple decision to either
        submit (99999) or cancel (66666) a message that will potentially lead
        to a crossed limit order book pre-match.
        Consequently, a matching step must follow. The idea is to "first add all
        liquidity and then take away matched
        liquidity".

        Submission orders have to be indicated with template_id = 99999.
        Cancellation messages have to be indicated with template_id = 66666.

        Orders submitted via update_with_agent_message() are stored in the
        actual internal state and can affect the market. When they are executed,
        they take liquidity of historic orders (market impact).

        :param message:
            dict, agent message
        :return trade_list:
            list, dicts with executed trades
        """
        self.agent_message_list.append(message)
        # update internal limit order book state ...
        #TODO: Should I just take the current self._state_timestamp as priority time for the message, i.e.
        # message['PriorityTime'] = self._state_timestamp (maybe + some latency effect...)

        # order submit
        if message["template_id"] == 99999:
            self._order_submit(message)
        # order cancel
        elif message["template_id"] == 66666:
            self._order_cancel(message)

        # match internal limit order book state
        # trade_list = self.match() # Old...
        trade_list = self.match_new(state_to_match=self._state) # New...
        #removes empty price levels from state
        self._clean_state()

        #DEBUGGING
        print('NEW MATCHING WITH SELF.STATE')
        print(trade_list)

        return trade_list

    def _order_submit(self, message):
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
        # TODO: remove .get() use [price]
        self._state[side].get(price, []).append(message)

        print(f'(INFO) Agent Order Submitted: Side: {side} | Price: {price} | Quantity: {quantity}')

    def _order_cancel(self, message):
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
            # TODO: idx not required...
            for idx, order in enumerate(self._state[side][price]):
                # identify order by timestep and remove order
                if order["timestamp"] == timestamp:
                    self._state[side][price].remove(order)
                    # log cancellation:
                    print(f'(INFO) Agent Order Cancelled: Side: {side} | Price: {price} | Timestamp: {timestamp}')

            for position, message in enumerate(self._state[side][price]):
                if message["timestamp"] == timestamp:
                    del self._state[side][price][position]  # TODO gucken ob nicht auch del obj funktioniert
                    # if no order left on price level, delete price level
                    if not self._state[side][price]:
                        del self._state[side][price]  # TODO gucken ob nicht auch del obj funktioniert
                    # TODO: wofür der return?
                    return True

        except Exception as error:
            print(error)

    def _order_amend(self):  # equivalent to 'cancel and resubmit'
        pass

    #  agent order simulation WITHOUT MARKET IMPACT . . . . . . . . . . . . . . . . . . . . . .

    # TODO: DEBUGGING
    #  prepare specific orders and check if they get executed as aspected
    #  check if agent_message_list gets updated correctly
    #  check if agent_trade_list gets updated correctly
    #  make sure that the internal state is not affected by the simulation
    def simulated_update_with_agent_message(self, message):
        """
        Process simulated agent messages:

        Each incoming agent message gets the current state timestamp plus latency as
        message timestamp.

        Submission messages (template_id=99999) get appended to the agent_message_list.
        Cancellation messages (template_id=66666) are executed by changing the template_id
        of the cancelled order message from 99999 to 33333. To cancel an order it has to be
        identifiable by the price, side, timestamp combination.

        Cancellation messages are also appended to the agent_message_list to have a complete
        protocol.

        Messages submitted via simulated_update_with_agent_message() are stored and matched
        separately from the internal state and do not affect the market (no market impact).

        :param message:
            dict, agent message
        :return:
        """
        # set current statetime plus latency as agent message timestamp
        latency = 10 #ns
        message['timestamp'] = self.timestamp + latency
        message['msg_seq_num'] = None
        message['agent_msg_num'] = len(self.agent_message_list)

        # Store the agent order to agent_message_list
        if message['template_id'] == 99999:
            self.agent_message_list.append(message)

        elif message['template_id'] == 66666:
            # append cancellation message to agent_message_list
            self.agent_message_list.append(message)
            price = message['price']
            side = message['price']
            timestamp = message['timestamp']
            # filter agent_message_list for agent_message to delete (called agent message to differ from incoming message)
            #TODO: use filter instead
            for agent_message in self.agent_message_list:
                if agent_message['price'] == price and agent_message['side'] == side \
                        and agent_message['timestamp'] == timestamp:
                    # change template_id of resting agent message from 99999 to 33333 to mark as CANCELLED
                    agent_message['template_id'] = 33333

        else:
            print('(WARNING) agent message template_id not valid.')
        #DEBUGGING
        #print('AGENT MESSAGE LIST')
        #print(self.agent_message_list)

        #FOR TESTING
        self.simulate_agent_order_matching()

    def simulate_agent_order_matching(self):
        """
        Method to match simulated agent orders with the internal state.
        The agent orders are stored as messages is agent_message_list and matched
        with simulation_state, a light version copy of the internal market state
        which only contains relevant price levels (for higher efficiency)
        """

        trade_list = [] #TODO: necessary?
        # dicts to create simulation_state
        simulation_state = {}
        bid_side_dict = {}
        ask_side_dict = {}
        # lists for values to calculate thresholds
        # (lists needed if agent messages exist only on one side)
        bid_threshold_values = []
        ask_threshold_values = []

        # -- define thresholds for relevant state (i.e. which price levels need to be present)

        # TODO: Should I put "build simulation state" part in a new helper method (übersichtlicher..?)
        # filter prices from agent messages
        buy_prices = []
        sell_prices = []

        for message in self.agent_message_list:
            # filter for order submissions (exclude cancellations):
            if message['template_id'] == 99999:
                if message['side'] == 1:
                    buy_prices.append(message['price'])
                if message['side'] == 2:
                    sell_prices.append(message['price'])

        # compute min/max prices
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

        # -- construct simulation_state from self._state using the relevant keys

        # store relevant bid levels to bid_side_dict
        # Note: According to my tests, copy is sufficient to not affect internal state
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

        # Note: messages can just be appended, they will be executed according to their priority time
        # Note: I copy the messages instead of giving a reference to the original message, this means that
        # masseges in the agent_message_list will not directyl be affected by matchin.
        for message in self.agent_message_list:

            if message['template_id'] == 99999:
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

        # -- match simulation_state, receive trade list (if orders could be executed)
        trade_list = self.match_new(state_to_match=simulation_state)
        # print('NEW MATCHING WITH SIMULATION STATE')
        print("TRADE LIST")
        print(trade_list)

        # -- update agent_message_list
        if trade_list:
            self._process_executed_agent_orders(trade_list)

        # -- store executed orders in a datastructure for agent-trades (e.g. TradeClass or Market.agent_trades...)
        if trade_list:
            self._store_agent_trades(trade_list)
            print("AGENT TRADE LIST", self.agent_trade_list)

    def _process_executed_agent_orders(self, trade_list):
        """
        Helper method to process executed agent messages in the agent_message_list.
        If an agent order is fully
        executed, its template_id gets updated to 11111. If an agent order
        is partially executed, the execution volume is deducted from the quantity
        and the order remains active (template_id = 99999). To keep track of
        partial executions, their volume is stored in 'partial executions'.

        :param trade_list
            list, contains execution summaries from match()
        """
        # iterate over trades
        for trade in trade_list:
            # agent_msg_num is unique identifier of message
            if 'agent_msg_num' in trade.keys():
                agent_msg_num = trade['agent_msg_num']
                executed_volume = trade['quantity']
                # TODO: actually relevant for Trade, but could also be saved in order...
                execution_price = trade['quantity']
                execution_time = trade['timestamp']

                # -- filter out the affected agent messages by message-id
                # Note: executed_order is reference to mutable message object -> message can be manipulated
                message = next(filter(lambda message: message['agent_msg_num'] == agent_msg_num, self.agent_message_list))

                # -- manipulate the agent message to process the execution

                # agent orders was fully executed (possibly after partial executions)
                if executed_volume >= message['quantity']:
                    message['template_id'] = 11111  # -11111: fully executed
                    if 'partial_executions' in message.keys():
                        message['partial_executions'].append(message['quantity'])

                # agent order was partially executed
                if executed_volume < message['quantity']:
                    # add partial execution indicator
                    if 'partial_executions' in message.keys():
                        message['partial_executions'].append(executed_volume)

                    else:
                        message['partial_executions'] = [executed_volume] # list
                        message['original_quantity'] = message['quantity']

                    # remove executed quantity
                    message['quantity'] = message['quantity'] - executed_volume


    def _store_agent_trades(self, trade_list):
        """
        Store agent trade summaries to agent_trade_list.
        :param trade_list,
            list, contains execution summaries from match()
        """
        for trade in trade_list:

            agent_trade = {}

            agent_trade['agent_trade_num'] = len(self.agent_trade_list)
            agent_trade['execution_time'] = trade['timestamp']
            agent_trade['executed_volume'] = trade['quantity']
            agent_trade['execution_price'] = trade['price']
            #TODO: agent order side is more relevant
            agent_trade['aggressor_side'] = trade['aggressor_side']
            agent_trade['agent_msg_num'] = trade['agent_msg_num']

            self.agent_trade_list.append(agent_trade)

    def _match_agent_orders_against_execution_summary(self, message_packet): # "arbeitstitel"
        """
        => Simuliere, ob die agent orders gegen market- bzw. marketable limit orders
        ausgeführt worden wären.
        => hilfreich: alle relevanten messages werden (idR) zusammen in einem message packet
        geliefert.
        => Should be called if template_id == 13202 - execution summary
        # -
        :return: trade_list
            list, trades as dictionaries
        """

        # -- filter out the execution summary message to get execution infos
        exec_sum_message = list(filter(lambda d: d['template_id'] == 13202, message_packet.copy()))[0]
        aggressor_side = exec_sum_message['side'] # "AggressorSide"
        # worst price of this match.
        last_price = exec_sum_message['price'] # LastPx
        aggressor_timestamp = exec_sum_message['timestamp'] # AggressorTime

        # -- get historically executed orders from message_packet
        executed_orders = list(filter(lambda d: d['template_id'] in [13104, 13105], message_packet.copy()))

        # -- get potentially executable agent orders from agent_message_list (side and ACTIVE - 99999)
        complementary_agent_orders = list(filter(lambda d: d['side'] != aggressor_side and d['template_id'] == 99999,
                                                 self.agent_message_list.copy()))
        # agent-ask <= aggressor bid:
        if aggressor_side == 1:
            executable_agent_orders = list(filter(lambda d: d['price'] <= last_price, complementary_agent_orders))
        # agent-bid >= aggressor ask:
        elif aggressor_side == 2:
            executable_agent_orders = list(filter(lambda d: d['price'] >= last_price, complementary_agent_orders))

        # sort executable_agent_orders by price and timestamp
        if aggressor_side == 2:
            # sort by highest price and smallest timestamp
            executable_agent_orders = sorted(executable_agent_orders, key=lambda d: (-d['price'], d['timestamp']))
        elif aggressor_side == 1:
            # sort by lowest price and smallest timestamp
            executable_agent_orders = sorted(executable_agent_orders, key=lambda d: (d['price'], d['timestamp']))

        # -- test if agent orders would have had a higher priority than historically executed orders
        trade_list = []
        ######
        print('executed_orders BEFORE: ', executed_orders)
        ######
        for agent_order in executable_agent_orders:

            match_execution_summary = None  # reset
            agent_msg_num = agent_order["agent_msg_num"]

            for executed_order in list(executed_orders): # list() to not mess up the iterator while removing elements
                ###
                print('next_order')
                ###
                # set flag to false
                agent_execution_possible = False

                execution_price = executed_order['price']  # for match_execution_summary
                execution_quantity = executed_order['quantity']  # for match_execution_summary

                if aggressor_side == 1 and agent_order['price'] < executed_order['price']:
                    agent_execution_possible = True

                elif aggressor_side == 2 and agent_order['price'] > executed_order['price']:
                    agent_execution_possible = True

                # smaller timestamps have higher priority
                elif agent_order['price'] == executed_order['price'] and agent_order['timestamp'] < executed_order['timestamp']:
                    agent_execution_possible = True

                if agent_execution_possible:
                    # create execution summary
                    match_execution_summary = {"aggressor_side": aggressor_side,
                                               "price": execution_price,
                                               "timestamp": aggressor_timestamp,
                                               "quantity": execution_quantity,
                                               "agent_msg_num": agent_msg_num}

                    # add to trade_list
                    trade_list.append(match_execution_summary)
                    # remove executed_order from executed_orders
                    executed_orders.remove(executed_order)
                    ######
                    print('executed_orders AFTER: ', executed_orders)
                    ######
        #DEBUGGING:
        print('AGENT ORDER EXECUTED AGAINST EXECUTION SUMMARY:')
        print('SPECIAL TRADE-LIST: ', trade_list)
        print('AGENT MESSAGE LIST BEFORE EXECUTION', self.agent_message_list)
        print('AGENT TRADE LIST BEFORE EXECUTION: ', self.agent_trade_list)

        # process agent executions
        if trade_list:
            self._process_executed_agent_orders(trade_list)
            self._store_agent_trades(trade_list)

        print('AGENT MESSAGE LIST AFTER EXECUTION', self.agent_message_list)
        print('AGENT TRADE LIST AFTER EXECUTION: ', self.agent_trade_list)

    # match . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    # Can be removed when match_new() is tested and debugged...
    def match(self):
        """
        Given the internal state (potentially crossed), match all relevant
        orders.
        """

        # assert that price level on either side exists!
        assert ...
        # ...
        trade_list = []

        while True:

            # get the top of the limit order, best bid (max_buy) and best ask (min_sell)
            max_buy = max(self._state[1].keys())  # best bid
            min_sell = min(self._state[2].keys())  # best ask

            # if state is not crossed, stop matching immediately
            if not max_buy >= min_sell:
                break

            # if empty, remove the current best price_level and continue with the loop
            if not (price_level := self._state[1][max_buy]):
                del self._state[1][max_buy];
                continue  # TODO gucken ob nicht auch del obj funktioniert
            # ...
            if not (price_level := self._state[2][min_sell]):
                del self._state[2][min_sell];
                continue  # TODO gucken ob nicht auch del obj funktioniert
            # in the following, match these two particular orders
            order_buy = self._state[1][max_buy][0]
            order_sell = self._state[2][min_sell][0]
            # assign roles for these two particular orders
            order_aggressor, order_standing = sorted([order_buy, order_sell],
                                                     key=lambda x: x["timestamp"], reverse=True,
                                                     )
            # ...
            delta = order_buy["quantity"] - order_sell["quantity"]

            # TODO: if order buy < oder sell:
            # (buy - sell) < 0: buy executed at min_sell, sell reduced to delta
            if delta < 0:
                order_sell["quantity"] -= order_buy["quantity"]
                trade_list.append({"aggressor_side": order_aggressor["side"],
                                   "price": order_sell["price"],  # min_sell
                                   "timestamp": order_aggressor["timestamp"],
                                   "quantity": order_buy["quantity"],
                                   })
                self._state[1][max_buy].remove(order_buy)  # TODO gucken ob nicht auch del funktioniert

            # (buy - sell) > 0: sell executed at max_buy, buy reduced to delta
            elif delta > 0:
                order_buy["quantity"] -= order_sell["quantity"]
                trade_list.append({"aggressor_side": order_aggressor["side"],
                                   "price": order_buy["price"],  # max_buy
                                   "timestamp": order_aggressor["timestamp"],
                                   "quantity": order_sell["quantity"],
                                   })
                self._state[2][min_sell].remove(order_sell)  # TODO gucken ob nicht auch del funktioniert

            # (buy - sell) = 0: buy and sell are executed
            else:
                trade_list.append({"aggressor_side": order_aggressor["side"],
                                   "price": order_standing["price"],
                                   "timestamp": order_aggressor["timestamp"],
                                   "quantity": order_buy["quantity"],  # = order_sell["quantity"]
                                   })
                self._state[1][max_buy].remove(order_buy)  # TODO gucken ob nicht auch del funktioniert
                self._state[2][min_sell].remove(order_sell)  # TODO gucken ob nicht auch del funktioniert

        # update timestamp afer matching
        # self._update_internal_timestamp()

        return trade_list

    # -> modified match method (08/Sept/2022)
    def match_new(self, state_to_match):
        """
        Idea: use the same matching method for simulated matching without market impact
        and "real" matching with market impact by passing the respective state
        (internal_state or simulation_state) as input argument.

        The simulated state is entirely copied and does not include references to the
        internal state or the agent_message_list. Hence, manipulating the simulation
        state does neither affect the internal state nor the agent_message_list. Instead,
        agent messages are executed separately in ...

        :param state_to_match,
            dict, state which should be matched (either simulation_state or self._state)
        """
        # check if state_to_match has two order book sides (simulation_state can sometimes be empty..)
        #TODO: test if condition works
        if state_to_match[1] and state_to_match[2]:

            trade_list = []

            # loop until best price levels cannot be matched
            while True:

                max_buy = max(state_to_match[1].keys())  # best bid
                min_sell = min(state_to_match[2].keys())  # best ask
                # DEBUGGING
                # print('max_buy: ', max_buy, 'min_sell: ', min_sell)

                # break if order book not crossed
                if not max_buy >= min_sell:
                    break

                # if empty, remove the current best price_level and continue with the loop
                if not (price_level := state_to_match[1][max_buy]):  # :=  allows variable assignment inside expression
                    del state_to_match[1][max_buy];
                    continue

                if not (price_level := state_to_match[2][min_sell]):
                    del state_to_match[2][min_sell];
                    continue

                # sort by priority time and select first order (lowest priority time), necessary for simulation!
                order_buy = sorted(state_to_match[1][max_buy], key=lambda d: d['timestamp'])[0]
                order_sell = sorted(state_to_match[2][min_sell], key=lambda d: d['timestamp'])[0]

                # aggressor order has later timestamp
                order_standing, order_aggressor = sorted([order_buy, order_sell], key=lambda x: x["timestamp"])

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

                # if agent-message was matched, add agent_msg_num
                if "agent_msg_num" in order_buy.keys():
                    match_execution_summary["agent_msg_num"] = order_buy["agent_msg_num"]
                elif "agent_msg_num" in order_sell.keys():
                    match_execution_summary["agent_msg_num"] = order_sell["agent_msg_num"]
                # TODO: Eigenausführungen verhindern?
                # Edge-Case, both sides are agent orders:
                elif "agent_msg_num" in order_buy.keys() and "agent_msg_num" in order_sell.keys():
                    match_execution_summary["agent_msg_num"] = list(order_sell["agent_msg_num"],
                                                                    order_buy["agent_msg_num"])
                else:
                    pass

                trade_list.append(match_execution_summary)

                # DEBUGGING
                # print(trade_list)

                return trade_list

        else:
            print('(WARNING) State is not complete - no matching possible')

    # TODO: what was the idea behind this string representation, specific use case?
    def __str__(self):
        """
        String Representation
        :return: string
            str, representation of class.
        """
        return json.dumps(self._state, sort_keys=True, indent=4)

    def to_json(self, filename):
        """
        Store current state to a json file.
        :param filename:
            str, name / directory of the file to store state in
        """
        with open(filename, 'w') as f:
            json.dump(self.state_l3, f)

    @classmethod
    def reset_instance(class_reference):
        """
        Reset class instance.
        """
        # delete class instance
        del class_reference.instance
