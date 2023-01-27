""" In order to have a baseline strategy for the agent, he should get small
incentives to learn the TWAP strategy and apply it if nothing else happens.
This is a agent prototype to develop and test this reward.

AGENTID = 211222
"""
__author__ = "florian"
__date__ = "2022-12-12"
__version__ = "0.1"

# TODO: Implement TWAP incentive.

from copy import copy

import numpy as np
import pandas as pd

from market.market import Market
from market.market_interface import MarketInterface
from context.context import Context
from feature_engineering.market_features import MarketFeatures
from reinforcement_learning.reward.abc_reward import BaseReward
from reinforcement_learning.observation_space.abc_observation_space \
    import BaseObservationSpace
from reinforcement_learning.base_agent.abc_base_agent import RlBaseAgent
from reinforcement_learning.transition.agent_transition import AgentTransition
from reinforcement_learning.action_space.action_storage import ActionStorage
from context.agent_context import AgentContext
from agent.agent_trade import AgentTrade
from agent.agent_order import OrderManagementSystem as OMS
from agent.agent_metrics import AgentMetrics
from reinforcement_learning.transition.env_transition import \
    EnvironmentTransition
from reinforcement_learning.action_space.abc_action_space import \
    BaseActionSpace
from market.market_metrics import MarketMetrics
from market.market_trade import MarketTrade


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
        # Raw market data.
        #raw_obs = self.raw_market_features
        # Hand-crafted market data.
        crafted_obs = self.handcrafted_market_features
        # Append raw and handcrafted market features.
        #market_obs = np.append(raw_obs, crafted_obs)

        return crafted_obs #market_obs

    def agent_observation(self) -> np.array:
        """
        Implement the agent observation.
        """
        # Use standard agent obs with elapsed time and remaining inventory.
        agent_obs = self.standard_agent_observation

        return agent_obs

    def holistic_observation(self):
        """
        Combine market_obs and agent_obs to one array which
        can be fed into the NN.
        """
        market_obs = self.market_observation()

        holistic_obs = market_obs
        holistic_obs.astype('float32')

        return holistic_obs


class Reward(BaseReward):
    """
    Subclass of BaseReward to implement the reward for a specific agent.
    The abc method receive_reward needs to be implemented.
    """

    def __init__(self, twap_n=10):
        super().__init__()
        # dynamic attributes
        self.number_of_trades = 0
        self.number_of_orders = 0

        initial_inventory = AgentContext.initial_inventory
        episode_length = AgentContext.episode_length
        self.twap_child_quantity = initial_inventory / twap_n
        self.twap_interval = episode_length / twap_n

    def receive_reward(self):
        """Define the Specific reward signal."""

        reward = self.immediate_absolute_is_reward
        # reward = self.terminal_absolute_is_reward
        # reward = self.incentivize_waiting(reward_factor=20)

        return reward


class ActionSpace(BaseActionSpace):
    """Specific Implementation of action space."""

    def __init__(self,
                 verbose=False,
                 num_twap_intervals=6):
        super().__init__()

        self.verbose = verbose
        self.num_twap_intervals = num_twap_intervals

        self.agent_metrics = AgentMetrics()
        self.market_features = MarketFeatures()

    def take_action(self,
                    action):
        """
        Take action as input and translate into trading decision.
        :param action,
            ..., next action
        """
        self.limit_and_qt_action(action)


class ISAgent2(RlBaseAgent):
    """
    Agent with a larger action space, e.g. selection between different
    limits and quantities.
    """

    def __init__(self,
                 initial_inventory: int = 800_0000,
                 verbose=False,
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
        # Store initial inventory to agent context.
        AgentContext.update_initial_inventory(self.initial_inventory)
        # Store Episode Length:
        AgentContext.update_episode_length_ns(episode_length)

        # DEBUGGING:
        # print("INITIAL INV", AgentContext.initial_inventory)

        self.verbose = verbose
        self.quantity = 10_0000
        # Convert episode_length to nanoseconds
        self.episode_length = pd.to_timedelta(episode_length).delta

        # dynamic
        self.first_step = True
        self.final_market_order_submitted = False

        # compositions
        self.market_interface = MarketInterface()
        self.reward = Reward()
        self.observation_space = ObservationSpace()
        self.market_features = MarketFeatures()
        self.agent_metrics = AgentMetrics()
        self.action_space = ActionSpace()

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
        Execute the given action.
        :param action,
            int, action
        """
        self.action_space.take_action(action)

    def reset(self):
        """Reset"""
        super().__init__()
        self.quantity = self.quantity
        self.verbose = self.verbose

        # compositions
        self.market_interface.reset()
        self.reward.reset()
        self.observation_space.reset()
        self.market_features.reset()
