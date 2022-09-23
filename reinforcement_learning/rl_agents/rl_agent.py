#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 21/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Prototype for RL Agents.
-----------------------
Everything is wrapped into an own RLAgent class for fast prototyping.

# - replay is responsible for done and info
# - RLAgent is responsible for take_action, observation, reward

#...
def replay.step(self, action)

    if self.step > (len(self.episode)):
        self.done = True

    observation, reward = RLAgent.step(action) => RLAgent

    info = {'pnl': AgentMetrics.pnl, 'num_trades':AgentMetrics.num_trades}

    return observation, reward, done, info

"""
# ---------------------------------------------------------------------------

from market.market_interface import MarketInterface
from feature_engineering.market_features import MarketFeatures
from reinforcement_learning.observation_space import ObservationSpace
from reinforcement_learning.reward import Reward

# TODO abc.ABC

class RlAgent:

    def __init__(self,
                 quantity: int = 10_0000):

        # static
        self.quantity = quantity

        # class instances
        self.market_interface = MarketInterface()
        self.market_features = MarketFeatures()
        self.reward = Reward()
        # should include market- and agent features
        self.observation_space = ObservationSpace()

    def step(self, action):

        print('level2')
        print(self.market_features.level_2())
        # take action
        self.take_action(action)

        # make observation
        observation = self.observation_space

        # receive reward
        reward = self.reward.pnl_marginal

        # return

        return observation, reward

    # ActionSpace
    def take_action(self, action):

        best_ask = self.market_features.best_ask()
        best_bid = self.market_features.best_bid()

        print(best_ask)
        print(best_bid)

        # buy
        if action == 1 and best_ask:
            self.market_interface.submit_order(side=1,
                                               limit=best_ask,
                                               quantity=self.quantity)
            print('buy submission')
        # sell
        elif action == 2 and best_bid:
            self.market_interface.submit_order(side=2,
                                               limit=best_bid,
                                               quantity=self.quantity)
            print('sell submission')
        # wait
        else:
            pass
