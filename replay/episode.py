#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Episode Module for the Level-3 backtest engine"""
# ---------------------------------------------------------------------------
import json
import os

import pandas as pd

from reconstruction.reconstruction import Reconstruction

# PATH TO DATA DIRECTORY
#PATH = "/Users/florianewald/PycharmProjects/Level3-Data-Analysis/sample_msg_data/DE0005190003.XETR_20220201T080007_20220201T120000"
#local = "/Users/florianewald/PycharmProjects/Level3-Data-Analysis/sample_msg_data/DE0005190003.XETR_20220201T120211_20220201T163000"

local = "/Users/florianewald/PycharmProjects/A7_data/"

PATH = local


class Episode:

    def __init__(self,
                 identifier:str = None,
                 episode_start = None,
                 episode_end = None):
        """

        """
        # static attributes from arguments
        self.identifier = identifier
        self.episode_start = episode_start
        self.episode_end = episode_end

        # dynamic attributes
        self._step = 0
        self._timestamp = None
        self.snapshot_end = None
        self.snapshot_start = None
        self.message_list = None
        self.message_packet_list = None

        # map stock ETR to ISIN
        self.isin_dict = {"BMW": "DE0005190003",
                        "FME": "DE0005785802"}


        self.load_data_new()



    def load_data_new(self):

        #TODO: epsiode_start/end besser als str oder datetime?

        ###FOR TESTING ###
        #self.episode_start = pd.Timestamp('2022-02-16 08:10:00+0000', tz='UTC')
        #self.episode_end = pd.Timestamp('2022-02-16 08:20:00+0000', tz='UTC')
        #################

        start_date_str = self.episode_start.strftime("%Y%m%d")
        start_hour = self.episode_start.strftime("%H")

        isin = self.isin_dict[self.identifier]
        market = ".XETR_"

        # NOTE: THIS IS FOR DATA FILES WHICH ARE SPLIT AT MID-AUCTION

        # trading_time
        # Note: 12 in UTC time (MEZ would be 13)
        if int(start_hour) <= 12:
            trading_time = "T08"
        elif int(start_hour) >= 12:
            trading_time = "T12"

        # create pattern
        pattern = isin + market + start_date_str + trading_time

        # TODO: just repeat until a valid basepath was found (with new dates/ids), but controlled from replay not from episode...?
        base_path = None
        # find base_path name
        for directory in os.listdir(PATH):
            if pattern in directory:
                print(directory)
                base_path = directory

        if not base_path:
            print("Base_Path not found")


        # load files from base_path
        snapshot_start_path = open(f"{PATH + base_path}/snapshot_start.json")
        snapshot_end_path = open(f"{PATH + base_path}/snapshot_end.json")
        message_list_path = open(f"{PATH + base_path}/message_list.json")

        # (start snapshots are sometimes in a list)
        snapshot_start = json.load(snapshot_start_path)[0]
        # TODO: not required..
        snapshot_end = json.load(snapshot_end_path)
        # slice out the first message (reconstruction message)
        message_packet_list = json.load(message_list_path)[1:]

        # instantiate reconstuction
        reconstruction = Reconstruction()
        # load state from snapshot_start
        reconstruction.initialize_state(snapshot=snapshot_start)

        # convert to unix
        episode_start_unix = (self.episode_start - pd.Timestamp(
            "1970-01-01", tz='UTC')) // pd.Timedelta('1ns')

        episode_end_unix = (self.episode_end - pd.Timestamp(
            "1970-01-01",tz='UTC')) // pd.Timedelta('1ns')

        # run reconstruction until episode start is reached
        for message_packet in message_packet_list:

            reconstruction.update_with_exchange_message(message_packet)

            if int(message_packet[0]['TransactTime']) > episode_start_unix:
                break

        # assert deviation
        assert reconstruction._state_timestamp - episode_start_unix < 6e10, \
            "Divergence at Episode Snapshot larger 1 Min"

        # set episode_start_snapshot
        self.episode_start_snapshot = reconstruction._state

        # filter message list for episode messages
        self.episode_message_list = list(filter(lambda p:
                            int(p[0]['TransactTime']) > episode_start_unix and
                            int(p[0]['TransactTime']) < episode_end_unix,
                            message_packet_list))

        # assert deviation
        if self.episode_message_list:
            assert abs(int(self.episode_message_list[0][0]
                           ['TransactTime']) - episode_start_unix
                       ) < 6e10, "Divergence at Episode Start larger 1 Min"

        if self.episode_message_list:
            assert abs(
                int(self.episode_message_list[-1][0]
                    ['TransactTime']) - episode_end_unix
                ) < 6e10, "Divergence at Episode Start larger 1 Min"

    # Preliminary Version to develop replay etc.
    def load_data(self): #OLD todo: rename "load_specific_files", can be useful for testing...
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

    # Note: Not Tested!
    def __iter__(self):
        """
        Create episode as iterable generator which can be
        used to step the backtest internally from replay.
        """
        for step, message_packet in enumerate(self.message_packet_list):

            next_message_packet = self.message_packet_list[step]

            yield next_message_packet

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

    def reset(self):
        pass