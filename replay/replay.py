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
#from .market_state_new import MarketState
from market.market_state_v1 import MarketStateAttribute
from agent.context import Context
from agent.market_interface import MarketInterface

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

        # instantiate market interface
        self.market_interface = MarketInterface()
        print(type(self.market_interface))

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

        #TODO: Do I have to call market_state.match() manually?
        MarketStateAttribute.instance.update_with_exchange_message(message_packet)

    #--- EXPERIMENTAL METHOD
    def _showcase_submit_and_cancel_order_impact(self): # delete method later...
        """
        Showcase of how the MarketInterface class can be used to submit
        and cancel orders.
        :return:
        """
        # -- get next message packet update
        next_message_packet = self.episode.__next__()
        # -- update market
        self._market_step(message_packet=next_message_packet)

        print('STEP: ', self.step_counter)
        # ... test order submission
        price = 9411000000
        ts = 1643716931606681391
        side = 1
        if self.step_counter == 0:
            self.market_interface.submit_order_impact(side=side, quantity=2220000, timestamp=ts, limit=price)

        # ... test order cancellation

        if self.step_counter == 10:
            message = {}
            # cancellation
            message["template_id"] = 66666
            message["side"] = side
            message["price"] = price
            message["timestamp"] = ts
            print('MESSAGE:', message)
            # directly
            # MarketStateAttribute.instance.update_with_agent_message(message)
            # via MarketInterface
            self.market_interface.cancel_order_impact(side=side, limit=price, timestamp=ts)


        # check state for my message/order
        if price in MarketStateAttribute.instance._state[1].keys():
            print(MarketStateAttribute.instance._state[1][price])
        # test order cancellation
        self.step_counter = self.step_counter + 1


    def step(self):
        # -- RL: done flag
        if self.episode._step >= (self.episode.__len__() - 1):
            self.done = True
            print('(ENV) DONE')

        # -- get next message packet update
        next_message_packet = self.episode.__next__()

        # -- update market
        self._market_step(message_packet=next_message_packet)

        # -- retrieve current state form MarketState:
        current_state = MarketStateAttribute.instance.state_l3

        # check if own trades are stored in the state
        # print(current_state[1])

        # -- update Context list
        Context(market_state=current_state)


        # ... buy: test order submission
        price = 9450000000
        ts = 1643716931606681391
        side = 1
        if self.step_counter == 0:
            self.market_interface.submit_order_impact(side=side, quantity=2220000, timestamp=ts, limit=price)

        price = 9413000000
        ts = 1643716931606681391
        side = 1
        if self.step_counter == 0:
            self.market_interface.submit_order(side=side, quantity=1110000, timestamp=ts, limit=price)

        # ... sell: test order submission
        price = 9418000000
        ts = 1643716931606681391
        side = 2
        if self.step_counter == 0:
            self.market_interface.submit_order(side=side, quantity=3330000, timestamp=ts, limit=price)

        price = 9421000000
        ts = 1643716931606681391
        side = 2
        if self.step_counter == 0:
            self.market_interface.submit_order(side=side, quantity=8880000, timestamp=ts, limit=price)
        # ... test order cancellation
        if self.step_counter == 10:

            # via MarketInterface
            self.market_interface.cancel_order(side=side, limit=price, timestamp=ts)


        # check state for my message/order
        #if price in MarketStateAttribute.instance._state[1].keys():
        #    print(MarketStateAttribute.instance._state[1][price])
        # test order cancellation
        self.step_counter = self.step_counter + 1


    def _reset_market_with_attributes(self, snapshot_start):
        # reset old market instance
        MarketStateAttribute.reset_instance()
        # create new instance of MarketState
        _ = MarketStateAttribute(market_id="ID_0")
        # build initial state of new MarektState instance from snapshot_start
        MarketStateAttribute.instance.initialize_state(snapshot=snapshot_start)

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
