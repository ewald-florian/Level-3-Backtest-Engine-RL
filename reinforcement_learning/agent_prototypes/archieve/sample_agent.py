#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-09-21"
__version__ = "0.1"
# ----------------------------------------------------------------------------
"""

***NOTE: THIS IS AN OLD VERSION OF AN SAMPLE AGENT!***
***FROM BEFORE THE CANGE TO THE ABSTRACT CLASS STRUCTURE***

Prototype No. 1 for RL Agents (SampleAgent.
--------------------------------------------------------------------
Idea: wrap competitive tasks in an abstract class agent

- replay is responsible for done and info (and Context)
- RLAgent is responsible for take_action, observation, reward+

There are three adjustment points for specialized agents:
1. Action: define how the agent takes actions
2. Observation: define the observation the agent should make (this has to
be done in ObservationSpace so far, will maybe sourced out, e.g. in an
abstract class structure or just normal inheritance...)
3. choose the reward the agent receives (implement different reward functions
inside the Reward class)

___________________________________________________________________________
Endo-to-End:

for step in number_of_steps:

    replay.step(self, action)
    
        if self.step > (len(self.episode)):
            self.done = True
    
        observation, reward = RLAgent.step(action) => RLAgent
    
        info = {'pnl': AgentMetrics.pnl, 'num_trades':AgentMetrics.num_trades}
    
        return observation, reward, done, info
___________________________________________________________________________
"""
from copy import copy

from market.market_interface import MarketInterface
from reinforcement_learning.observation_space.observation_space import ObservationSpace
from reinforcement_learning.reward.reward import Reward
from feature_engineering.market_features import MarketFeatures


class RlAgent:

    def __init__(self,
                 quantity: int = 10000_0000,
                 verbose=True):

        # static
        self.quantity = quantity
        self.verbose = verbose

        # class instances
        self.market_interface = MarketInterface()
        self.market_features = MarketFeatures()
        self.reward = Reward()
        # should include market- and agent features
        self.observation_space = ObservationSpace()

    def step(self, action):

        # take action
        self.take_action(action)

        # make observation
        observation = self.observation_space.holistic_observation()

        # receive reward
        reward = copy(self.reward.pnl_realized) # property
        #print('(SAMPLE AGENT) reward: ', reward)

        # return

        return observation, reward

    # ActionSpace
    def take_action(self, action):

        # submit markatable limit orders
        best_ask = self.market_features.best_ask()
        best_bid = self.market_features.best_bid()

        # buy
        if action == 1 and best_ask:
            self.market_interface.submit_order(side=1,
                                               limit=best_ask,
                                               quantity=self.quantity)
            if self.verbose:
                print('(RL AGENT) buy submission: ', best_ask)
        # sell
        elif action == 2 and best_bid:
            self.market_interface.submit_order(side=2,
                                               limit=best_bid,
                                               quantity=self.quantity)
            if self.verbose:
                print('(RL AGENT) sell submission: ', best_bid)
        # wait
        else:
            if self.verbose:
                print('(RL AGENT) wait')
