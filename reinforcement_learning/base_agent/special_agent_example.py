#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Example for the usage of the Template for RL-Agent Prototypes.
--------------------------------
A Prototype of a 'Special Agent' has to be build using a construct of
abstract- and subclasses. The SpecialAgent is then controlled from replay.

This is a simple PnL trader as example.

- ObservationSpace subclasses BaseObservationSpace, the abstract methods which
    need to be implemented are agent_observation and market_observation.
- Reward subclasses BaseReward, the abstract method which needs to be
    implemented is receive_reward.
- SpecialAgent subclasses RLBaseAgent, the abstract methods which need to
    be implemented are step, _take_action and reset.
- SpecialAgent builds compositions of ObservationSpace and Reward for RL and
    MarketInterface to execute actions.

The first goal of this architecture is to allow fast efficient prototyping. The
sole requirement for a new prototype is the respective implementation of the
abstract methods. The second goal is sustainable archiving of prototypes. Each
prototype can be stored in one piece as a single py file containing the
implemented subclasses ObservationSpace, Reward and >SpecialAgent<.

This template can be copied to implement a new prototype. Prototypes should
have a unique and meaningful name and be located in the agent_prototypes
package.
"""

from copy import copy

import numpy as np

from market.market_interface import MarketInterface
from feature_engineering.market_features import MarketFeatures
from reinforcement_learning.reward.abc_reward import BaseReward
from reinforcement_learning.observation_space.abc_observation_space \
    import BaseObservationSpace
from reinforcement_learning.base_agent.abc_base_agent import RlBaseAgent
from reinforcement_learning.transition.agent_transition import AgentTransition
from reinforcement_learning.action_space.action_storage import ActionStorage


class ObservationSpace(BaseObservationSpace):
    """
    Subclass of BaseObservationSpace to implement the observation for a
    specific agent. The abstract methods market_observation and
    agent_observation need to be implemented.
    """
    def __init__(self):
        """
        Initiate parent class via super function.
        """
        super().__init__()

    def market_observation(self) -> np.array:
        """
        Implement the market observation.
        """
        # -- market features
        market_obs = self.market_features.level_2_plus(store_timestamp=False,
                                                  data_structure='array')


        if market_obs is not None:
            prices = market_obs[::3]
            quantities = market_obs[1::3]
            # -- normalize
            prices = self._min_max_norma_prices_clipped(prices)
            quantities = self._min_max_norma_quantities_clipped(quantities)
            market_obs[::3] = prices
            market_obs[1::3] = quantities

        return market_obs

    def agent_observation(self) -> np.array:
        """
        Implement the agent observation.
        """
        return np.array([])


class Reward(BaseReward):
    """
    Subclass of base reward to implement the reward for specific agent.
    The abc method receive_reward needs to be implemented.
    """
    def __init__(self):
        super().__init__()

    def receive_reward(self):
        reward = self.pnl_realized
        return reward


class SpecialAgent(RlBaseAgent):
    """
    Template for SpecialAgents which are based on specific reward and
    observation space.
    """
    def __init__(self,
                 quantity: int = 10000_0000,
                 verbose=True
                 ):
        """
        When initialized, SpecialAgent builds compositions of MarketInterface,
        Reward and ObservationSpace. Note that Reward and ObservationSpace
        are subclasses which should be implemented to meet the specific
        requirements of this special agent, a specific observation and a
        specific reward function.
        """
        # static
        self.quantity = quantity
        self.verbose = verbose

        # compositions
        self.market_interface = MarketInterface()
        self.reward = Reward()
        self.observation_space = ObservationSpace()

        self.market_features = MarketFeatures()

    def step(self):
        """
        Step executes the action, gets a new observation, receives the reward
        and returns reward and observation.
        """
        # get action from ActionStorage
        action = ActionStorage.action
        #print('(AGENT) action ', action)
        self._take_action(action)

        observation = copy(self.observation_space.holistic_observation())
        reward = copy(self.reward.receive_reward())
        #print('(AGENT) reward: ', reward)
        #print('(AGENT) observation: ', observation)

        # pass obs, reward to Env via AgentTransition.transition
        AgentTransition(observation, reward)
        #print('(AGENT) AgentTransition: ', AgentTransition.transition)

    def _take_action(self, action):
        """
        Take action as input and translate into trading decision.
        :param action,
            ..., next action
        """
        # submit marketable limit orders
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

    def reset(self):
        super().__init__()
        self.quantity = self.quantity
        self.verbose = self.verbose

        # compositions
        self.market_interface.reset()
        self.reward.reset()
        self.observation_space.reset()
        self.market_features.reset()

