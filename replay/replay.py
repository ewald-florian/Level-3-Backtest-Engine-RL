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
from agent.context import Context
from agent.market_interface import MarketInterface
from agent.agent_trade import AgentTrade
from market.market_trade import MarketTrade
from agent.agent_order import OrderManagementSystem as OMS
from agent.agent_metrics import AgentMetrics
from agent.observation_space import ObservationSpace

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

    def _market_step(self, message_packet):

        Market.instances['ID'].update_with_exchange_message(message_packet)

    def step(self):

        # -- RL: done flag
        if self.episode._step >= (self.episode.__len__() - 1):
            self.done = True
        # -- get next message packet update
        next_message_packet = self.episode.__next__()
        # -- update market
        self._market_step(message_packet=next_message_packet)
        # -- retrieve current state form MarketState:
        # TODO: test without L3 next (this is even without levels...)!
        current_state = Market.instances['ID'].state_l3
        # -- append state to Context
        Context(market_state=current_state)
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        print('OBS', ObservationSpace.market_observation())

######################  test order submission etc  #####################################################



        """
        price = 9418000000
        side = 1
        if self.step_counter % 10 == 0:
            MarketInterface.submit_order(side=side, quantity=2220000, limit=price)
            print('INFO(SUBMISSION)')

        # ... test order cancellation
        if self.step_counter == 100:
            # via MarketInterface
            MarketInterface.cancel_order(order_message_id=0)
            print('INFO(CANCELLATION)')

        price = 9413000000
        side = 2
        if self.step_counter % 20 == 0:
            MarketInterface.submit_order(side=side, quantity=2220000, limit=price)
            print('INFO(SUBMISSION)')
        """

####################################################################################################
        self.step_counter = self.step_counter + 1


    def _reset_market_with_attributes(self, snapshot_start):
        # reset old market instance
        Market.reset_instances()
        # create new instance of MarketState
        _ = Market(market_id="ID")
        # build initial state of new MarektState instance from snapshot_start
        Market.instances['ID'].initialize_state(snapshot=snapshot_start)

        # instantiate agent metrics
        _ = AgentMetrics()

    def _reset_context(self):
        # reset context
        Context.reset_context_list()

    def _reset_market_interface(self):
        # not necessarily needed
        pass

    def reset(self):
        self._build_new_episode()
        # episode has to build before _reset_market to provide new snapshot_start
        # --- marekt_state as instance of replay
        #self._reset_market(self.episode.snapshot_start)

        # -- market state as independent class attribute
        self._reset_market_with_attributes(snapshot_start=self.episode.snapshot_start)

        self._reset_context()


