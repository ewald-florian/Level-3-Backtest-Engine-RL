#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal blueprint RL-Agent, customizable to different strategies.
"""

from copy import copy

import numpy as np
import pandas as pd

from market.market_interface import MarketInterface
from episode.episode import Episode
from feature_engineering.market_features import MarketFeatures
from reinforcement_learning.reward.abc_reward import BaseReward
from reinforcement_learning.observation_space.abc_observation_space \
    import BaseObservationSpace
from reinforcement_learning.base_agent.abc_base_agent import RlBaseAgent
from reinforcement_learning.transition.agent_transition import AgentTransition
from reinforcement_learning.action_space.action_storage import ActionStorage
from context.agent_context import AgentContext
from agent.agent_metrics import AgentMetrics
from reinforcement_learning.action_space.abc_action_space import \
    BaseActionSpace
from utils.initial_inventory import initial_inventory_dict


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
        pass

    def agent_observation(self) -> np.array:
        """
        Implement the agent observation.
        """
        pass


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
        pass


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
        pass


class FinalOEAgent1(RlBaseAgent):
    """
    Agent with a larger action space, e.g. selection between different
    limits and quantities.
    """

    def __init__(self,
                 initial_inventory_level: str = "Avg-10s-Vol",
                 verbose=False,
                 episode_length="10s",
                 ):
        """
        Initialize Blueprint.
        """
        self.initial_inventory_level = initial_inventory_level
        # Get initial inventory from initial_inventory_dict.
        self.initial_inventory = initial_inventory_dict[
            Episode.current_identifier][self.initial_inventory_level]*1_0000
        # Store initial inventory to agent context.
        AgentContext.update_initial_inventory(self.initial_inventory)
        # Store Episode Length:
        AgentContext.update_episode_length_ns(episode_length)

        self.verbose = verbose
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
        pass

    def reset(self):
        """Reset"""
        super().__init__()

        self.verbose = self.verbose

        # compositions
        self.market_interface.reset()
        self.reward.reset()
        self.observation_space.reset()
        self.market_features.reset()
