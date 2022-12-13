#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
__author__ = "phillipp"
__date__ = "?"
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Reconstruction Class. Reconstructs Market State from proprietary A7 EOBI Data.
"""
# ---------------------------------------------------------------------------


# TODO: Theoretisch müsste man auch mit der reconstruction message (erste
#  message im message_packet (bzw. am morgen) den snapshot_start herstellen
#  können. -> ausprobieren
import json
import copy
import pandas as pd

from reconstruction.parser import MessagePacketParser
from reconstruction.parser import SnapshotParser


class Reconstruction:
    """
    Reconstruction class is used to reconstruct the market state from
    proprietary A7 EOBI message data (Eurex, Xetra, EEX und CME).

    First, market state has to be initialized with a starting snapshot using
    the initialize_state method.

    Subsequently, the message data can be processed packet for packet where
    the individual messages are parsed and the respective changes at the state
    are conducted. This can be done using the update_with_exchange_message
    method.
    """

    def __init__(self,
                 track_timestamp: bool = True,
                 track_index: bool = True):
        """
        State is implemented as a stateful object that reflects the current
        limit order book snapshot in full depth and full detail.

        The state consists of a dictionary with two sides, which have the
        integer keys 1 (=Buy) and 2 (=Sell). Each of
        these sides contains a dictionary with price levels. The keys for the
        price levels are the prices as integers.
        Since the structure of a dictionary does not require the keys to be in
        order, the price levels in the dict may
        not be in order. Each price level contains a list of orders, which are
        in price-time-order. The orders are
        represented as dicts with the following values: "template_id",
        "msg_seq_num", "side", "price", "quantity"
        and "timestamp". All orders must contain the same six values and all
        of these values must be integers:

        Possible values:

       "template_id":  13100 (order_add), 13101 (order_modify), 13102
                        (order_delete), 13103 (order_mass_delete), 13104
                        (execution_full), 13105 (execution_partial),
                        13106 (order_modify_same_priority), 13202
                        (execution_summary), 99999 (order_submit agent-side),
                        66666 (order_cancel agent-side)
        "msg_seq_num":  any integer, but must be in ascending order
        "side": 1 (buy side) or 2 (sell side)
        "price":    any integer, that is a valid pirce for the security. Note
                    that the price is always represented as an integer, e.g.
                    123.45 would be 12345000000 with 8 decimals.
        "quantity": any integer, that is a valid quantity for the security
                    with 4 decimal, i.e. 10 would be 10000.
        "timestamp":   an integer that represents a unix timestamp

        :param track_timestamp
            bool, if True the internal timestamp will be tracked
        :param track_index
            bool, if True the internal index will be tracked
        ...
        """

        # -- static attributes from input arguments
        self.track_timestamp = track_timestamp
        self.track_index = track_index

        # -- dynamic attributes
        # internal state
        self._state = None
        # use 'msg_seq_num' to validate exchange-based updates
        self._state_index = None
        # use 'timestamp' to validate agent-based update
        self._state_timestamp = None

    # start/end snapshot . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @SnapshotParser.parse
    def initialize_state(self, snapshot):
        """
        Set current state to be the provided snapshot from which to start
        reconstruction.

        The original snapshot format is parsed by SnapshotParser
        (as decorator).

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
        self._initialize_internal_state_index()
        # set initial timestamp
        self._initialize_internal_timestamp()

        # logging
        print("(INFO) State Build from Snapshot: Start Timestamp: {}".format(
            pd.to_datetime(self._state_timestamp, unit='ns')))

    def initialize_state_from_parsed_snapshot(self, snapshot):
        """
        Episode class generates start snapshot with  reconstruction. Hence,
        these snapshots are already parsed and can be loaded without the
        @SnapshotParser.parse decorator.

        Set current state to be the provided snapshot from which to start
        reconstruction.

        The original snapshot format is parsed by SnapshotParser
        (as decorator).

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
        self._initialize_internal_state_index()
        # set initial timestamp
        self._initialize_internal_timestamp()

        # logging
        print("(INFO) State Build from Snapshot: Start Timestamp: {}".format(
            pd.to_datetime(self._state_timestamp, unit='ns')))

    @SnapshotParser.parse
    def validate_state(self, snapshot) -> bool:
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
            # store copy of keys by using list()
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

    '''
    # NEW METHOD (MATCH instead of IF-CONDITIONS)
    # WARNING: ONLY WORKS WITH PYTHON 3.10 OR HIGHER
    @MessagePacketParser.parse
    def update_with_exchange_message_new(self, message_packet) -> list:
        """
        Updates are based on exchange-based message packet, a set of messages
        that describe in detail each change to be made to the limit order book
        post-match (`_order_add`, `_order_delete`, ...). The idea is to "add
        only unmatched liquidity at each time step".

        The messages are parsed to a format similar to:
        {'template_id': X, 'msg_seq_num': X, 'side': X, 'price': X,
        'quantity': X, 'timestamp': X, 'time-in': X}

        - time-in: TrdRegTSTimeIn

        Concrete format of parsed message depends on message type
        (template_id), see MessagePacketParser class for more information.

        :param message_packet:
            list, contains single parsed messages as dicts.
        """

        # assert that self._state exists
        assert self._state, \
            "(ERROR) note that an update cannot take place without self._state"

        # iterate over all messages in message_packet
        for message in message_packet:

            # assert that self._state_index is smaller than 'msg_seq_num'
            assert self._state_index <= message["msg_seq_num"], \
                "(ERROR) update requires 'msg_seq_num' larger than index"

            # -- update internal limit order book state ...
            match message["template_id"]:
                # order add
                case 13100:
                    self._order_add(message)
                # order delete
                case 13102:
                    self._order_delete(message)
                # order delete mass
                case 13103:
                    self._order_delete_mass()
                # order modify
                case 13101:
                    self._order_modify(message)
                # order modify same priority
                case 13106:
                    self._order_modify_same_priority(message)
                # execution full
                case 13104:
                    self._execution_full(message)
                # execution partial
                case 13105:
                    self._execution_partial(message)
                # execution summary
                # Note: Execution Summaries are processed in Market
                #   Market Trades are stored to MarketTrade.history.
                case 13202:
                    pass
                case _:
                    pass

        # -- update internal timestamp
        if self.track_timestamp:
            self._update_internal_timestamp(message_packet)

        # -- update internal index
        if self.track_index:
            self._update_internal_state_index(message_packet)

        # Note: parsed message_packet is used in Market
        return message_packet
    '''

    @MessagePacketParser.parse
    def update_with_exchange_message(self, message_packet) -> list:
        """
        Updates are based on exchange-based message packet, a set of messages
        that describe in detail each change to be made to the limit order book
        post-match (`_order_add`, `_order_delete`, ...). The idea is to "add
        only unmatched liquidity at each time step".

        The messages are parsed to a format similar to:
        {'template_id': X, 'msg_seq_num': X, 'side': X, 'price': X,
        'quantity': X, 'timestamp': X, 'time-in': X}

        - time-in: TrdRegTSTimeIn

        Concrete format of parsed message depends on message type
        (template_id), see MessagePacketParser class for more information.

        :param message_packet:
            list, contains single messages as dicts, message types can be
            inferred from TemplateIDs
        """

        # assert that self._state exists
        assert self._state, \
            "(ERROR) note that an update cannot take place without self._state"

        # iterate over all messages in message_packet
        for message in message_packet:

            # assert that self._state_index is smaller than 'msg_seq_num'
            # plus some tolerance
            assert self._state_index <= (message["msg_seq_num"] + 100), \
                "(ERROR) update requires 'msg_seq_num' larger than index. " \
                "state_index: {} message_index: {} message: {}".format(
                    self._state_index, message["msg_seq_num"], message)

            # -- update internal limit order book state ...

            # order add
            if message["template_id"] == 13100:
                self._order_add(message)
            # order delete
            elif message["template_id"] == 13102:
                status = self._order_delete(message)
            # order delete mass
            elif message["template_id"] == 13103:
                self._order_delete_mass()
            # order modify
            elif message["template_id"] == 13101:
                self._order_modify(message)
            # order modify same priority
            elif message["template_id"] == 13106:
                self._order_modify_same_priority(message)

            # execution full
            elif message["template_id"] == 13104:
                    self._execution_full(message)

            # execution partial
            elif message["template_id"] == 13105:
                    self._execution_partial(message)

            # execution summary
            elif message["template_id"] == 13202:
                pass

            else:
                pass

        # -- update internal timestamp
        if self.track_timestamp:
            self._update_internal_timestamp(message_packet)

        # -- update internal index
        if self.track_index:
            self._update_internal_state_index(message_packet)

        # Note: parsed message_packet is used in Market
        return message_packet

    # helper functions to update state from messages . . . . . . . . . . . . .

    def _initialize_internal_timestamp(self):
        """
        Initialized the internal timestamp afer snapshot is parsed.
        Searches for highest timestamp in the entire state.
        """
        max_timestamp = 0
        for side in self._state.keys():
            for price_level in self._state[side].keys():
                for order in self._state[side][price_level]:
                    if (order["timestamp"]) > max_timestamp:
                        max_timestamp = order["timestamp"]

        # max timestamp existent in current _state
        self._state_timestamp = max_timestamp

    def _initialize_internal_state_index(self):
        """
        Initialized internal state index after starting snapshot is
        parsed. Searches for highest message sequence number in the
        entire state.
        """
        max_index = 0

        # search state for highest 'msg_seq_num'
        for side in self._state.keys():
            for price_level in self._state[side].keys():
                for order in self._state[side][price_level]:
                    # Agent Masseges have None as msg_seq_num...
                    if order["msg_seq_num"]:
                        if (order["msg_seq_num"]) > max_index:
                            max_index = order["msg_seq_num"]

        self._state_index = max_index

    def _update_internal_timestamp(self, message_packet):
        """
        Update internal timestamp from latest message packet.
        """
        max_timestamp = None
        # relevant messages are order_add, order_modify (and execution_summary)
        relevant_messages = list(filter(lambda d: d['template_id'] in
                                                  [13100, 13101, 13105],
                                                    message_packet))

        if relevant_messages:
            max_timestamp = max(d["timestamp"] for d in relevant_messages)
        # update internal timestamp
        if max_timestamp:
            self._state_timestamp = max_timestamp

    def _update_internal_state_index(self, massage_packet):
        """
        Update internal state index from latest message packet.
        Equals largest msg_seq_num.
        """

        # filter out header messages (13005, 13004 etc.)
        valid_messages = list(filter(lambda d: d['msg_seq_num'] != None,
                                     massage_packet))
        # get max
        if valid_messages:
            max_msg_seq_num = max(d['msg_seq_num'] for d in valid_messages)

        self._state_index = max_msg_seq_num

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

        # TODO: use filter!
        # note: sometimes flawed messages with non-existent price levels
        try:
            for position, message in enumerate(self._state[side][price_level]):
                # search for the order with the specific timestamp
                if message["timestamp"] == timestamp:
                    del self._state[side][price_level][position]
                    # if no order left on price level, delete price level
                    if not self._state[side][price_level]:
                        del self._state[side][price_level]
        except:
            pass

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
        # order delete template_id
        message_prev["template_id"] = 13102  # order delete

        self._order_delete(message_prev)

        # add order with message_keep (last queue position)
        message_keep = {key: message[key] for key in [
            "template_id", "msg_seq_num", "side",  # shared
            "price", "quantity", "timestamp",
        ]}

        message_keep["template_id"] = 13101  # order modify

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

        try:
            for position, message in enumerate(self._state[side][price_level]):
                if message["timestamp"] == timestamp:
                    self._state[side][price_level][position] = message_keep
        except:
            pass

    def _execution_full(self, message):  # 13104
        """
        Extract trade information from execution full message (13104)
        and delete respective order from state.
        :param message:
            dict, message
        """
        # delete order if order is fully executed
        self._order_delete(message)

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
        """
        # extract message information
        side = message["side"]
        price_level = message["price"]
        timestamp = message["timestamp"]
        quantity = message["quantity"]  # executed quantity

        # note: sometimes flawed messages with non-existent price levels
        try:
            # iterate over specified price level
            for position, message in enumerate(self._state[side][price_level]):
                if message["timestamp"] == timestamp:
                    # subtract executed quantity
                    self._state[side][price_level][position][
                        "quantity"] -= quantity
                    # if order has quantity 0, delete order
                    if self._state[side][price_level][position]["quantity"] == 0:
                        self._order_delete(message)
        except:
            pass

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