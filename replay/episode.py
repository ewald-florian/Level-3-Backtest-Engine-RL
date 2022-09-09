#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Episode Module for the Level-3 backtest engine"""
# ---------------------------------------------------------------------------

import pandas as pd
import json

# PATH TO DATA DIRECTORY
#PATH = "/Users/florianewald/PycharmProjects/Level3-Data-Analysis/sample_msg_data/DE0005190003.XETR_20220201T080007_20220201T120000"
PATH = "/Users/florianewald/PycharmProjects/Level3-Data-Analysis/sample_msg_data/DE0005190003.XETR_20220201T120211_20220201T163000"
class Episode:

    def __init__(self):
        """

        :rtype: object
        """
        # load data
        _ = self.load_data()
        self._step = 0
        self._timestamp = 0

    # Preliminary Version to develop replay etc.
    def load_data(self):
        '''
        Load data
        '''
        snapshot_start_file = open(f"{PATH}/snapshot_start.json")
        snapshot_end_file = open(f"{PATH}/snapshot_end.json")
        message_list_file = open(f"{PATH}/message_list.json")

        #TODO: why is snapshot start in a list?
        self.snapshot_start = json.load(snapshot_start_file)[0]
        self.snapshot_end = json.load(snapshot_end_file)
        # slice out the first message (reconstruction message)
        self.message_packet_list = json.load(message_list_file)[1:]

    def generate_episode(self):
        #TODO:
        # - generate episodes from episode start list
        # - translate start_point and end_point in UNIX time
        # - be able to generate a start_snapshot for any given start time
        # - (optional) end_snapshot for any given end time
        pass

    def __next__(self):
        '''
        Returns next message_packet and counts episode steps.
        __next__ can be called to step the episode externally,
        for example in a RL training loop.
        :return message_packet
            list, contains messages as dictionaries.
        '''
        # retrieve next message packet
        next_message_packet = self.message_packet_list[self._step]
        # TODO: Which timestamp as global timestamp?
        # update timestamp with message-header TransactTime
        self._timestamp = next_message_packet[0]['TransactTime']
        # count step
        self._step += 1
        # return message_packet
        return next_message_packet

    def __iter__(self):
        pass

    def __len__(self):
        '''
        Returns length of the current episode, relevant for training loops
        over episodes and the done-flag of the RL environment. Based on
        length of the message_packet_list.
        :return: current_episode_length
            int, length of episode
        '''
        # TODO: What would be appropiate as length, message_packets or single messages?
        current_episode_length = len(self.message_packet_list)

        return current_episode_length

    @property
    def timestamp(self):
        """
        UTCTimestamp: Timestamps are in UTC, and represented as nanoseconds past the UNIX epoch
        (00:00:00 UTC on 1 January 1970).
        :return: timestamp, int
        """
        return self._timestamp

    @property
    def timestamp_datetime(self):
        """
        Timestamp converted to datetime.
        :return: timestamp, datetime
        """
        utct_timestamp = int(self._timestamp)
        datetime_timestamp = pd.to_datetime(utct_timestamp, unit='ns')
        return datetime_timestamp

    @property
    def episode_start(self):
        pass

    @property
    def episode_end(self):
        pass