#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : florian
# version ='1.0'
# ---------------------------------------------------------------------------
""" Episode Module for the Level-3 backtest engine"""
# ---------------------------------------------------------------------------
import json
import os
import copy
import pickle
import platform

import pandas as pd

from reconstruction.reconstruction import Reconstruction

# Note: min max prices are stored for normalization in RL
from reinforcement_learning.observation_space.minmaxvalues \
    import MinMaxValues
from context.agent_context import AgentContext

# PATH TO DATA DIRECTORY
# old dataset
# local="/Users/florianewald/PycharmProjects/A7_data/"
# local new sample with Bayer january and february 2021
local="/Users/florianewald/PycharmProjects/A7_NEW_SAMPLE/"
# dataset on server
server_new = "/home/jovyan/_shared_storage/temp/A7_data/messages/"
# TODO: nvme and server path
# new dataset on nvme
# nvme = "/Volumes/WD_BLUE_NVME/A7_DATA/"

if platform.system() == 'Darwin':  # macos
    PATH = local
elif platform.system() == 'Linux':
    PATH = server_new
else:
    raise Exception('Specify PATH in episode.py')


class Episode:

    """
    Episode is the data storage for one episode in the backtest. It consists of
    a snapshot_start (used to initialize the market state in the beginning of
    the episode) and a message_packet_list (used to iteratively update the
    market state from messages and hence replay_episode the historical
    continuous trading phases.

    Data is given in UTC. XETRA trading hours reach from 08:00 to 16:30 UTC.
    The mid-action takes place from 12:00 to 12:02 UTC.

    Note: self.snapshot_start contains a snapshot which is already parsed since
    it was generated by the reconstruction class.
    """

    def __init__(self,
                 identifier:str = None,
                 episode_start = None,
                 episode_end = None,
                 verbose = True):
        """
        Generates episode by loading the respective start_snapshot and
        message_packet_list from the database  and storing them under the
        instance attributes self.start_snapshot and self.message_packet_list.

        Can be called from replay_episode to build the next episode for the
        backtest.

        :param identifier
            str, identifier of symbol
        :param episode_start,
            pd.Timestamp, start timestamp of the episode
        :param episode_end,
            pd.Timestamp, end timestamp of the episode
        :param verbose,
            if True print infos,
        """

        # static attributes from arguments
        self.identifier = identifier
        self.episode_start = episode_start
        self.episode_end = episode_end
        self.verbose = verbose

        # dynamic attributes
        self._step = 0
        self._timestamp = None
        self.snapshot_end = None
        self.snapshot_start = None
        self.message_packet_list = None
        self.episode_start_unix = None
        self.episode_end_unix = None

        # map stock ETR to ISIN
        self.isin_dict = {"BMW": "DE0005190003",
                        "FME": "DE0005785802",
                        # BAYER
                        "BAY": "DE000BAY0017",
                        # TELEKOM
                        "DTE": "DE0005557508",
                        # SAP
                        "SAP": "DE0007164600",
                        # ALLIANZ
                        "ALV": "DE0008404005",
                        # LINDE
                        "LIN": "IE00BZ12WP82"}

        # max quantities for normalization (90% quantile)
        self.max_qt_dict = {'BAY': 3878,
                            'SAP': 3024,
                            'LIN': 2657,
                            'ALV': 1790,
                            'DTE': 25256}

        # -- instantiate reconstruction
        # instantiate reconstuction composition
        self.reconstruction = Reconstruction()

        # -- load episode
        self.load_episode_data()

    def load_episode_data(self):
        """
        Use the given parameters to find the path to the respective dataset
        and load the data. Generate a start-snapshot for the beginning
        of the episode by reconstructing the trading period until that
        exact moment and store in self.snapshot_start. Slice out the message
        packages which occur during the episode and store them to
        self.message_packet_list. It both snapshot start and message list
        could be obtained, the new episode was successfully build.
        """

        start_date_str = self.episode_start.strftime("%Y%m%d")
        start_hour = self.episode_start.strftime("%H")

        isin = self.isin_dict[self.identifier]
        market = ".XETR_"

        # NOTE: THIS IS FOR DATA FILES WHICH ARE SPLIT AT MID-AUCTION
        # (...T08/T07... -> morning, ...T12/T11... -> afternoon)

        # trading_time (Note: 12:00 UTC, CET would be 13))

        # Winter time until 2021-03-28.
        if int(start_date_str) < 20210329:
            if int(start_hour) < 12:
                trading_time = "T08"
            elif int(start_hour) >= 12:
                trading_time = "T12"
        # Summer time from 2021-03-29.
        elif int(start_date_str) >= 20210329:
            if int(start_hour) < 11:
                trading_time = "T07"
            elif int(start_hour) >= 11:
                trading_time = "T11"

        # TODO: load data of days with volatility interruption

        # create pattern
        pattern = isin + market + start_date_str + trading_time

        # find base_path name
        base_path = None
        # The base_path contains directories for each isin which contain data.
        isin_path = PATH + isin + "/"

        for directory in os.listdir(isin_path):

            if pattern in directory:
                base_path = directory

        if not base_path and self.verbose:
            print("...Base_Path not found")

        # load files from base_path
        snapshot_start_path = open(f"{isin_path + base_path}/snapshot_start.json")
        #snapshot_end_path = open(f"{isin_path + base_path}/snapshot_end.json")
        message_list_path = open(f"{isin_path + base_path}/message_list.json")

        # (start snapshots are sometimes in a list)
        snapshot_start = json.load(snapshot_start_path)[0]
        #snapshot_end = json.load(snapshot_end_path)
        # slice out the first message (reconstruction message)
        message_packet_list = json.load(message_list_path)[1:]

        # load state from snapshot_start
        self.reconstruction.initialize_state(snapshot=snapshot_start)

        # convert to unix
        self.episode_start_unix = (self.episode_start - pd.Timestamp(
            "1970-01-01", tz='UTC')) // pd.Timedelta('1ns')

        self.episode_end_unix = (self.episode_end - pd.Timestamp(
            "1970-01-01",tz='UTC')) // pd.Timedelta('1ns')

        # run reconstruction until episode start is reached
        for message_packet in message_packet_list:

            self.reconstruction.update_with_exchange_message(message_packet)

            if int(message_packet[0]['TransactTime'])>self.episode_start_unix:
                break

        # assert deviation
        assert (self.reconstruction._state_timestamp - self.episode_start_unix
                ) < 6e10, "Divergence at Episode Snapshot larger 1 Min"

        # set episode_start_snapshot
        self.snapshot_start = copy.deepcopy(self.reconstruction._state)
        # faster: pickle.loads(pickle.dumps(self.reconstruction._state, -1))

        # Store level-20 bid/ask as initial min/max prices for normalization
        if self.snapshot_start:
            # prices
            buy_prices = list(self.snapshot_start[1].keys())
            buy_prices.sort(reverse=True)
            level_20_buy = buy_prices[20]
            MinMaxValues.update_min_price(level_20_buy)
            sell_prices = list(self.snapshot_start[2].keys())
            sell_prices.sort(reverse=False)
            level_20_sell = sell_prices[20]
            MinMaxValues.update_max_price(level_20_sell)
            # quantities (select max-qt from dict)
            asset_max_qt = self.max_qt_dict[self.identifier]
            MinMaxValues.update_max_qt(asset_max_qt)
            # Get start_time, end_time and episode duration for AgentContex
            AgentContext.update_start_time(self.episode_start_unix)
            AgentContext.update_end_time(self.episode_end_unix)
            episode_duration = self.episode_end_unix - self.episode_start_unix
            AgentContext.update_episode_length(episode_duration)

        # filter message list for episode messages
        self.message_packet_list = list(filter(lambda p:
                    int(p[0]['TransactTime']) > self.episode_start_unix and
                    int(p[0]['TransactTime']) < self.episode_end_unix,
                    copy.deepcopy(message_packet_list)))

        # assert deviation
        if self.message_packet_list:
            assert abs(int(self.message_packet_list[0][0]
                           ['TransactTime']) - self.episode_start_unix
                       ) < 6e10, "Divergence at Episode Start larger 1 Min"

        if self.message_packet_list:
            assert abs(
                int(self.message_packet_list[-1][0]
                    ['TransactTime']) - self.episode_end_unix
                ) < 6e10, "Divergence at Episode End larger 1 Min"

        # Assert if message list is long enough
        assert len(self.message_packet_list) > 0, "Message List Empty"

        # -- free up interpreter memory
        del snapshot_start
        del message_packet_list
        if 'snapshot_end' in locals():
            del snapshot_end
        del self.reconstruction._state

    # note: for development uses
    def load_specific_files(self, base_path: str):
        '''
        Load episode from specific base_path
        '''

        snapshot_start_file = open(f"{base_path}/snapshot_start.json")
        snapshot_end_file = open(f"{base_path}/snapshot_end.json")
        message_list_file = open(f"{base_path}/message_list.json")

        self.snapshot_start = json.load(snapshot_start_file)[0]
        self.snapshot_end = json.load(snapshot_end_file)
        self.message_packet_list = json.load(message_list_file)[1:]

    def __next__(self) -> list:
        '''
        Returns next message_packet and counts episode steps.
        __next__ can be called to step the episode externally,
        for example in a RL training loop.

        :return message_packet
            list, contains messages as dictionaries.
        '''

        # retrieve next message packet
        next_message_packet = self.message_packet_list[self._step]
        # count step
        self._step += 1
        # return message_packet
        return next_message_packet

    # Note: Not Tested!
    def __iter__(self):
        """
        Create episode as iterable generator which can be
        used to step the backtest internally from replay_episode.
        """
        for step, message_packet in enumerate(self.message_packet_list):

            next_message_packet = self.message_packet_list[step]

            yield next_message_packet

    def __len__(self) -> int:
        '''
        Returns length of the current episode as number of message packets,
        relevant for training loops over episodes and the done-flag of the RL
        environment. Based on length of the message_packet_list.

        :return: current_episode_length
            int, length of episode
        '''
        current_episode_length = len(self.message_packet_list)
        return current_episode_length

    def reset(self):
        pass