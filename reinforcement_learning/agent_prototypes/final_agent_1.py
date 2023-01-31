"""
This file contains the implementation of the final "main" optimal execution
agent of the master thesis:

- Full observation space (49)
- Full action space (12)

AGENTID = 290123
"""

from copy import copy

import numpy as np
import pandas as pd

from market.market import Market
from market.market_interface import MarketInterface
from context.context import Context
from replay_episode.episode import Episode
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
        # Raw market data.
        raw_obs = self.raw_market_features
        # Hand-crafted market data.
        crafted_obs = self.handcrafted_market_features
        # Append raw and handcrafted market features.
        market_obs = np.append(raw_obs, crafted_obs)

        return market_obs

    def agent_observation(self) -> np.array:
        """
        Implement the agent observation.
        """
        # Use standard agent obs with elapsed time and remaining inventory.
        agent_obs = self.standard_agent_observation

        return agent_obs


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

        #reward = self.immediate_absolute_is_reward()
        #reward = self.incentivize_waiting()
        #reward = self.terminal_absolute_is_reward()
        # For pretraining.
        #reward = self.incentivize_waiting(reward_factor=5)
        # For stabilizing the model in later stages.
        #reward = self.incentivize_waiting(reward_factor=0.0001)
        reward = self.twap_time_incentive_reward()

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
        When initialized, Agent builds compositions of MarketInterface,
        Reward and ObservationSpace. Note that Reward and ObservationSpace
        are subclasses which should be implemented to meet the specific
        requirements of this special agent, a specific observation and a
        specific reward function.

        Using the identifier which is stored in Episode.current_identifier,
        the Agent can access the asset-specific initial inventory in the
        initial_inventory_dict (utils). Thereby the initial_inventory_level
        must be given as argument.
        """
        self.initial_inventory_level = initial_inventory_level
        # Get initial inventory from initial_inventory_dict.
        self.initial_inventory = initial_inventory_dict[
            Episode.current_identifier][self.initial_inventory_level]*1_0000
        # Store initial inventory to agent context.
        AgentContext.update_initial_inventory(self.initial_inventory)
        # Store Episode Length:
        AgentContext.update_episode_length_ns(episode_length)

        # DEBUGGING:
        # print("INITIAL INV", AgentContext.initial_inventory)

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
        self.action_space.take_action(action)

    def reset(self):
        """Reset"""
        super().__init__()

        # Get initial inventory from initial_inventory_dict.
        self.initial_inventory = initial_inventory_dict[
            Episode.current_identifier][self.initial_inventory_level]*1_0000

        self.verbose = self.verbose

        # compositions
        self.market_interface.reset()
        self.reward.reset()
        self.observation_space.reset()
        self.market_features.reset()
