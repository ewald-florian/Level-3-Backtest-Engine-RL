"""Test Time and Inventory RL agent

match agent to loop:
AGENTID = 161122 # -> just use the date as agent ID

Idea to solve the variable problem:

Agent stores variables to AgentContext class attributes
AgentFeatures can access AgentContext to compute stuff
ObservationSpace can access AgentFeatures
Agent can access ObservationSpace to get obs.

Observation
-----------
Agent-Observation:
- normalized remaining inventory
- normalized remaining time
Market-Observation
- 5 levels of normalized LOB (Level 2+)

Action
------
- Sell or wait, fixed quantity, marketable limit order

Reward
------
"""
__author__ = "florian"
__date__ = "2022-11-13"
__version__ = "0.1"

from copy import copy

import numpy as np
import pandas as pd

from market.market import Market
from market.market_interface import MarketInterface
from feature_engineering.market_features import MarketFeatures
from reinforcement_learning.reward.abc_reward import BaseReward
from reinforcement_learning.observation_space.abc_observation_space \
    import BaseObservationSpace
from reinforcement_learning.base_agent.abc_base_agent import RlBaseAgent
from reinforcement_learning.transition.agent_transition import AgentTransition
from reinforcement_learning.action_space.action_storage import ActionStorage
from context.agent_context import AgentContext
from agent.agent_trade import AgentTrade


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
        # Use standard agent obs with elapsed time and remaining inventory.
        agent_obs = self.standard_agent_observation
        print("AgentObs: ", agent_obs)

        return agent_obs


class Reward(BaseReward):
    """
    Subclass of BaseReward to implement the reward for a specific agent.
    The abc method receive_reward needs to be implemented.
    """
    def __init__(self):
        super().__init__()
        # dynamic attributes
        #self.number_of_trades = 0

    def receive_reward(self):
        # TODO: I use the IS reward taken from is_agent
        # set is to 0
        latest_trade_is = 0
        # Check if new trades occured.
        num_new_trades = len(AgentTrade.history) - self.number_of_trades
        if num_new_trades:
            # Get volume-weighted latest_trade_is from agent_metrics method.
            latest_trade_is = self.agent_metrics.latest_trade_is(number_of_latest_trades=num_new_trades)
            # Update class intern trade counter.
            self.number_of_trades = len(AgentTrade.history)

        # return is as reward
        reward = latest_trade_is
        print(reward)
        return reward


class TimeInventoryAgent1(RlBaseAgent):
    """
    Template for SpecialAgents which are based on specific reward and
    observation space.
    """
    def __init__(self,
                 initial_inventory: int = 10000_0000,
                 verbose=True,
                 episode_length="1m",
                 ):
        """
        When initialized, SpecialAgent builds compositions of MarketInterface,
        Reward and ObservationSpace. Note that Reward and ObservationSpace
        are subclasses which should be implemented to meet the specific
        requirements of this special agent, a specific observation and a
        specific reward function.
        """
        # static
        self.initial_inventory = initial_inventory
        self.verbose = verbose
        self.quantity = 10_0000
        # Convert episode_length to nanoseconds
        self.episode_length = pd.to_timedelta(episode_length).delta

        # dynamic
        self.first_step = True

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
        self._take_action(action)

        observation = copy(self.observation_space.holistic_observation())
        reward = copy(self.reward.receive_reward())

        # pass obs, reward to Env via AgentTransition.transition
        AgentTransition(observation, reward)

    def _take_action(self, action):
        """
        Take action as input and translate into trading decision.
        :param action,
            ..., next action
        """
        # submit marketable limit orders
        best_ask = self.market_features.best_ask()
        best_bid = self.market_features.best_bid()

        # sell
        if action == 1 and best_ask:
            self.market_interface.submit_order(side=2,
                                               limit=best_bid,
                                               quantity=self.quantity)
            if self.verbose:
                print('(RL AGENT) buy submission: ', best_bid)

        # wait
        else:  # action == 0
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
