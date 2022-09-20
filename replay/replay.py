#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Replay Module for the Level-3 backtest engine"""
# ---------------------------------------------------------------------------

import random
import datetime

#!pip install pandas-market-calendars
import pandas_market_calendars as mcal
import pandas as pd

import logging
logging.getLogger().setLevel(logging.INFO)

from replay.episode import Episode
from market.market import Market
from market.context import Context
from agent.agent_metrics import AgentMetrics


# TODO: clean implementation after all classes are finished
# TODO: will also be responsible for passing input arguments to all the
# classes (i.e. latency to Market, tc_factor to AgentMetrics etc.), hence
# these methods cannot all be static since these input argements will be
# stored in replay.self...
#TODO: modes to run episode as list of dates or for a continuing time period

class Replay:

    # DE0005190003.XETR_20220201T080007_20220201T120000

    def __init__(self,
                 identifier_list:list=None,
                 identifier: str="BMW",
                 start_date:str="2022-02-01",
                 end_date:str="2022-02-02",
                 episode_length:str = "10m",
                 frequency:str="1m",
                 seed:int=None,
                 shuffle:bool=True,
                 random_identifier:bool=False,
                 exclude_high_activity_time:bool=False,
                 mode:str="random_episodes"):

        # -- static attributes
        self.identifier_list = identifier_list
        self.identifier = identifier
        self.random_identifier = random_identifier
        self.start_date = start_date
        self.end_date = end_date
        self.episode_length = episode_length
        self.frequency = frequency
        self.seed = seed
        self.shuffle = shuffle
        self.exclude_high_activity_time = exclude_high_activity_time
        self.mode = mode

        # -- dynamic attributes
        self.episode = None
        self.episode_start_list = []
        self.episode_counter = 0
        self.episode_index = 0
        self.step_counter = 0
        self.done = False

        # -- generate new episode_start_list
        self._generate_episode_start_list()

    # note: method is tested and debugged (20-09-22)
    def _generate_episode_start_list(self):
        """
        Generates a new episode start list which is a list of timestamps that
        can potentially be used as episode starting points.

        Epsiode_start_list only contains timestamps which are in XETRA
        trading hours and account for the episode length such that the time
        between episode_start and mid- or close auction is always adequate to
        run an entire episode.

        High activity trading times defined as 08:00-08:15 UTC and 16:15-16:30
        UTC can be optionally excluded using the exclude_high_activity_time
        argument.

        The episode list can optionally be shuffled using the shuffle argument,
        it is possible to define a specific seed using the seed argumen.
        """

        # xetra trading calendar (trading hours, weekends, holydays)
        xetr = mcal.get_calendar('XETR')
        schedule = xetr.schedule(start_date=self.start_date,
                                 end_date=self.end_date)

        # create date range with respective frequency
        self.episode_start_list = list(
            mcal.date_range(schedule, frequency=self.frequency))

        # -- account for market close and mid auction buffers
        # market_close (16:30 UTC)
        market_close = '01/01/10 16:30:00'  # arbitrary date
        # account for market close
        market_close = datetime.datetime.strptime(market_close,
                                                  '%m/%d/%y %H:%M:%S')

        # account for episode_length
        market_close_buffer = market_close - pd.Timedelta(self.episode_length)
        # mid auction (12.00 UTC)

        mid_auction = '01/01/10 12:00:00'  # arbitrary date

        mid_auction = datetime.datetime.strptime(mid_auction,
                                                 '%m/%d/%y %H:%M:%S')
        # account for episode_length
        mid_auction_buffer = mid_auction - pd.Timedelta(self.episode_length)

        # filter for close buffer
        self.episode_start_list = list(
            filter(lambda ts: (ts.time() < market_close_buffer.time()),
                   self.episode_start_list))

        # filter for mid auction buffer
        self.episode_start_list = list(
            filter(lambda ts: (ts.time() > mid_auction.time() or
                               ts.time() < mid_auction_buffer.time()),
                   self.episode_start_list))

        # -- exclude high trading activity time (optional)
        if self.exclude_high_activity_time:

            # high trading activity time in the beginning of the trading day
            high_activity_open = '01/01/10 08:15:00'  # arbitrary date
            # account for market close
            high_activity_open = datetime.datetime.strptime(
                high_activity_open, '%m/%d/%y %H:%M:%S')

            # high trading activity time in the end of the trading day
            high_activity_close = '01/01/10 16:15:00'  # arbitrary date
            # account for market close
            high_activity_close = datetime.datetime.strptime(
                high_activity_close, '%m/%d/%y %H:%M:%S')

            # filter
            self.episode_start_list = list(
                filter(lambda ts: (ts.time() > high_activity_open.time() and
                                   ts.time() < high_activity_close.time()),
                                    self.episode_start_list))

        # -- shuffle episode (optional
        if self.seed:
            random.seed(self.seed)

        if self.shuffle:
            random.shuffle(self.episode_start_list)

        # -- reset episode_counter to 0
        self.episode_counter = 0
        self.episode_index = 0

    # note: method is tested and debugged (20-09-22)
    def build_new_episode(self):
        """
        Use the next start timestamp in episode_start_list to build the next
        episode. It is not necessarily the case that each start_point in
        episode_start_list is equipped with the necessary data files, hence
        the construction of the new episode might fail. In this case, the
        episode is build with the next start timestamp until an episode can be
        build successfully.

        episiode_counter counts the successfully built episodes.
        episode_index counts the number of trials to build episodes
        and is therefore used to index the episode_start_list.

        A new episode is directly stored in the self.episode attribute.
        """

        # -- update episode parameters

        episode_start = self.episode_start_list[self.episode_index]
        episode_end = episode_start + pd.Timedelta(self.episode_length)

        if self.random_identifier and self.identifier_list:
            identifier = random.choice(self.identifier_list)
        else:
            identifier = self.identifier

        # -- build new episode

        for attempt in range(100):

            try:
                self.episode = Episode(episode_start=episode_start,
                                       episode_end=episode_end,
                                       identifier=identifier)

                # update episode_counter/index
                self.episode_counter += 1
                self.episode_index += 1

                print("(INFO) new episode was build in attempt: {}".format(
                    attempt+1))

            # return if episode could not be generated
            except:

                print("(ERROR) could not build episode from start_point")
                # update episode_counter

                self.episode_index += 1

                # update episode parameters
                episode_start = self.episode_start_list[self.episode_index]
                episode_end = episode_start + pd.Timedelta(self.episode_length)
                continue

            break

    # externally stepped replay . . . . . . . . . . . . . . . . . . . . . . .

    def _market_step(self):
        """
        Helper method to conduct the crucial market update operations:
        1. check if episode is done
        2. Receive new message packet from episode.__next__
        3. Pass new message packet to Market
        """

        # -- check if episode is done
        if self.episode._step >= (self.episode.__len__() - 1):
            self.done = True

        # -- update market with new message packet from episode
        message_packet = self.episode.__next__()

        # -- pass nex message packet to Market
        Market.instances['ID'].update_simulation_with_exchange_message(
            message_packet)

    def step(self):
        """
        External Stepping:
        -----------------
        Central Method to step the backtesting engine externally based
        on the episode __next__() method. Can be called from an external
        iterative training loop, for example in the "step" method of an
        RL-environment class in Open-AI Gym convention.
        """

        # market step
        self._market_step()
################## DEVELOPMENT AREA ##########################################
        state_l3 = Market.instances['ID'].state_l3




##############################################################################
        self.step_counter = self.step_counter + 1

    # internally stepped replay . . . . . . . . . . . . . . . . . . . . . . .

    def run_backtest(self):
        """"
        Internal Stepping:
        -----------------
        Central method to step the backtesting library internally. Based
        on episode __iter__ method.
        """
        pass

    # reset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    """
    Note: Replay is used as the entry point of the RL environment class
    to the backtest-library. Therefore, replay contains a universal reset
    method which not only resets replay but all backtest-engine classes when
    a new episodes starts. Usually, the RL-environment calls replay.reset()
    to reset the environment and replay.step() to step the environment.
    """

    def _reset_market(self, snapshot_start):
        # reset old market instance
        Market.reset_instances()
        # create new instance of MarketState
        _ = Market(market_id="ID")
        # build initial state of new MarektState instance from snapshot_start

        # -- from parsed snapshot (episode delivers parsed snapshots)
        # note: parsed snapshots have "template_id" instead of "Template_ID"
        some_price = list(snapshot_start[1].keys())[0]
        if "template_id" in snapshot_start[1][some_price][0]:
            Market.instances['ID'].initialize_state_from_parsed_snapshot(
                snapshot=snapshot_start)

        # -- from un-parsed snapshot (raw snapshot from database)
        else:
            Market.instances['ID'].initialize_state(snapshot=snapshot_start)

    @staticmethod
    def _reset_agent_metrics():
        # instantiate agent metrics
        _ = AgentMetrics()

    @staticmethod
    def _reset_context():
        # reset context
        Context.reset_context_list()

    def _reset_market_interface(self):
        # not necessarily needed
        pass

    def _reset_replay(self):

        self.build_new_episode()

    # TODO: call all reset helper methods!
    def reset(self):
        # episode has to be build before _reset_market
        # to provide new snapshot_start to Market
        self._reset_replay()

        self._reset_market(self.episode.snapshot_start)

        # -- market state as independent class attribute
        self._reset_market(snapshot_start=self.episode.snapshot_start)

        self._reset_context()


