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
import orjson
import json

# PATH TO DATA DIRECTORY
#PATH = "/Users/florianewald/PycharmProjects/Level3-Data-Analysis/sample_msg_data/DE0005190003.XETR_20220201T080007_20220201T120000"
local = "/Users/florianewald/PycharmProjects/Level3-Data-Analysis/sample_msg_data/DE0005190003.XETR_20220201T120211_20220201T163000"

PATH = local

class Episode:

    def __init__(self):
                 #symbol_list = None,
                 #episode_start = None,
                 #episode_end = None):
        """
        ...
        :rtype: object
        """

        # dynamic attributes
        self._step = 0
        self._timestamp = 0
        self.snapshot_end = []
        self.snapshot_start = []
        self.message_list = []
        self.message_packet_list = []

        # load data
        _ = self.load_data()

    # Preliminary Version to develop replay etc.
    def load_data(self):
        '''
        Load data
        '''

        snapshot_start_file = open(f"{PATH}/snapshot_start.json")
        snapshot_end_file = open(f"{PATH}/snapshot_end.json")
        message_list_file = open(f"{PATH}/message_list.json")

        
        self.snapshot_start = json.load(snapshot_start_file)[0]
        self.snapshot_end = json.load(snapshot_end_file)
        # slice out the first message (reconstruction message)
        self.message_packet_list = json.load(message_list_file)[1:]
        '''
        # Note: orjson is slightly faster than json
        # but requires adjustemnts
        snapshot_start_file = f"{PATH}/snapshot_start.json"
        snapshot_end_file = f"{PATH}/snapshot_end.json"
        message_list_file = f"{PATH}/message_list.json"

        with open(snapshot_start_file, "rb") as f:
            self.snapshot_start = orjson.loads(f.read())

        with open(snapshot_end_file, "rb") as f:
            self.snapshot_end = orjson.loads(f.read())

        with open(message_list_file, "rb") as f:
            self.message_list = orjson.loads(f.read())
        '''

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
        # count step
        self._step += 1
        # return message_packet
        return next_message_packet

    def __iter__(self):
        pass

    def __len__(self):
        '''
        Returns length of the current episode as number of message packets,
        relevant for training loops over episodes and the done-flag of the RL
        environment. Based on length of the message_packet_list.
        :return: current_episode_length
            int, length of episode
        '''
        current_episode_length = len(self.message_packet_list)
        return current_episode_length

    @property
    def episode_start(self):
        pass

    @property
    def episode_end(self):
        pass