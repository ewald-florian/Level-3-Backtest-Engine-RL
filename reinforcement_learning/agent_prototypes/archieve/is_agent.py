#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Agent based on implementation shortfall.

IS = side * (execution_price-arrival_price) / arrival_price

see Raja Velu p. 337.
"""
#----------------------------------------------------------------------------
__author__ = 'florian'
__date__ = '2022-11-05'
__version__ = '0.1'
# ---------------------------------------------------------------------------
#TODO: reset agent afer each episode... (to be called inside replay)
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

        # TODO: added this to avoid some import errors
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


# TODO: include implementation_shortfall into reward!
class Reward(BaseReward):
    """
    Subclass of base reward to implement the reward for specific agent.
    The abc method receive_reward needs to be implemented.
    """
    def __init__(self):
        super().__init__()

        # dynamic attributes
        self.number_of_trades = 0

    # TODO: move finished is-reward into BaseReward later and only call it here
    # TODO: "marginal is": 0 for no trade, IS of last trade for trade
    #  kann man über len(trade_list) machen, immer wenn sich diese verändert
    #  ähnlich wie bei PnL (dort war es halt easy weil ich einfach diff nehmen konnte)
    def receive_reward(self):

        # set is to 0
        latest_trade_is = 0
        new_number_of_trades = 0
        # only if there is a trade list already
        if self.agent_metrics.get_realized_trades:
            new_number_of_trades = len(self.agent_metrics.get_realized_trades)

        # update is only if there is a new trade (sparse reward)
        if new_number_of_trades > self.number_of_trades:
            latest_trade_is = self.agent_metrics.latest_trade_is
        # return is as reward
        reward = latest_trade_is
        print("IS reward", reward)
        return reward


class ISAgent(RlBaseAgent):
    """
    Template for SpecialAgents which are based on specific reward and
    observation space.
    """
    def __init__(self,
                 quantity: int = 10_0000,
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
            self.market_interface.submit_order(side=1, #1
                                               limit=best_ask,
                                               quantity=self.quantity)
            if self.verbose:
                print('(RL AGENT) buy submission: ', best_ask)
        # sell
        elif action == 2 and best_bid:
            self.market_interface.submit_order(side=2, #2
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

