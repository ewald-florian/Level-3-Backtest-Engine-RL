"""Test Time and Inventory RL agent

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
- 5 levels of normalized LOB


Action
------
- Sell or wait, fixed quantity, marketable limit order

Reward
------
"""

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
            prices = self._min_max_norma_prices(prices)
            quantities = self._min_max_norma_quantities(quantities)
            market_obs[::3] = prices
            market_obs[1::3] = quantities

        return market_obs

    def agent_observation(self) -> np.array:
        """
        Implement the agent observation.
        """
        # Note: very common agent state, see e.g. Beling/Liu
        time = self.agent_features.elapsed_time
        inv = self.agent_features.remaining_inventory
        agent_obs = np.array([time, inv])
        #DEBUGGING
        print("ObservationSpace agent_obs:", agent_obs)
        return agent_obs


class Reward(BaseReward):
    """
    Subclass of BaseReward to implement the reward for a specific agent.
    The abc method receive_reward needs to be implemented.
    """
    def __init__(self):
        super().__init__()

    def receive_reward(self):
        reward = self.pnl_realized
        return reward


class TimeInventoryAgent1(RlBaseAgent):
    """
    Template for SpecialAgents which are based on specific reward and
    observation space.
    """
    def __init__(self,
                 initial_inventory: int = 10000_0000,
                 verbose=True,
                 episode_length = "1m",
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
        # **NEW PART FOR AGENT OBSERVATION**
        # -- TIME and INITIAL INVENTORY
        # Get timestamp from Market in Unix format.
        current_time = Market.instances['ID'].timestamp
        # define start and end-time in the first episode
        if self.first_step:
            start_time = current_time
            # DEBUGGING
            print("CT", current_time)
            print("EL", self.episode_length)
            end_time = current_time + self.episode_length
            self.first_step = False
            print("start_time: ", start_time)
            print("end_time: ", end_time)
            # Add start_time, end_time, time_delta to AgentContext
            AgentContext.update_start_time(start_time=start_time)
            AgentContext.update_end_time(end_time)
            AgentContext.update_episode_length(self.episode_length)
            # Add initial inventory to AgentContext
            AgentContext.update_initial_inventory(self.initial_inventory)

            # DEBUGGING
            print("AC:")
            print("start_time", AgentContext.start_time)
            print("end_time", AgentContext.end_time)
            print("episode_length", AgentContext.episode_length)
            print("initial_inventory", AgentContext.initial_inventory)

        # get action from ActionStorage
        action = ActionStorage.action
        #print('(AGENT) action ', action)
        self._take_action(action)

        observation = copy(self.observation_space.holistic_observation())
        #DEBUGGING
        print("AGENT observation: ", observation)
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

        # sell
        if action == 1 and best_ask:
            self.market_interface.submit_order(side=2,
                                               limit=best_bid,
                                               quantity=self.quantity)
            if self.verbose:
                print('(RL AGENT) buy submission: ', best_bid)

        # also sell! since this is a prototype for execution agents
        elif action == 2 and best_bid:
            if action == 1 and best_ask:
                self.market_interface.submit_order(side=2,
                                                   limit=best_bid,
                                                   quantity=self.quantity)
            if self.verbose:
                print('(RL AGENT) buy submission: ', best_bid)
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
