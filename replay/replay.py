#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Replay Module for the Level-3 backtest engine"""
# ---------------------------------------------------------------------------

import logging
logging.getLogger().setLevel(logging.INFO)

from replay.episode import Episode
from market.market import Market
from market.context import Context
from agent.agent_metrics import AgentMetrics


class Replay:

    def __init__(self):
        # -- static attributes
        # self.start_date etc.

        # -- dynamic attributes
        self.episode_start_list = []
        self.episode_counter = 0
        self.episode_index = 0
        self.step_counter = 0
        self.done = False

        self.agent_metrics = AgentMetrics()


    def _generate_episode_start_list(self):
        pass

    def _build_new_episode(self):
        #TODO: add all options...
        try:
            self.episode = Episode()
            # update episode counter
            self.episode_counter += 1
            print("(INFO) new episode was build")
        except:
            print("(ERROR) could not build episode with the specified parameters")
            # also update episode_counter
            self.episode_counter += 1

        # RL: set done-flag for new episode false
        self.done = False
        # episode_index counts episodes which have been build successfully
        self.episode_index += 1

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
        Market.instances['ID'].update_with_exchange_message(message_packet)

        # -- check if episode is done
        if self.episode._step >= (self.episode.__len__() - 1):
            self.done = True

    def step(self):
        """
        External Stepping:
        -----------------
        Central Method to step the backtesting engine externally based
        on the episode __next__() method. Can be called from an external
        iterative training loop, for example in the "step" method of an
        RL-environment class in Open-AI Gym convention.
        """

        # conduct market step
        self._market_step()
################## DEVELOPMENT AREA ##########################################
        state_l3 = Market.instances['ID'].state_l3




##############################################################################
        self.step_counter = self.step_counter + 1

    # externally stepped replay . . . . . . . . . . . . . . . . . . . . . . .

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

    # TODO: clean implementation after all classes are finished
    #TODO: will also be responsible for passing input arguments to all the
    # classes (i.e. latency to Market, tc_factor to AgentMetrics etc.), hence
    # these methods cannot all be static since these input argements will be
    # stored in replay.self...
    #
    def _reset_market(self, snapshot_start):
        # reset old market instance
        Market.reset_instances()
        # create new instance of MarketState
        _ = Market(market_id="ID")
        # build initial state of new MarektState instance from snapshot_start
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

    # TODO: call all reset helper methods!
    def reset(self):
        # episode has to be build before _reset_market
        # to provide new snapshot_start to Market
        self._build_new_episode()

        self._reset_market(self.episode.snapshot_start)

        # -- market state as independent class attribute
        self._reset_market(snapshot_start=self.episode.snapshot_start)

        self._reset_context()


