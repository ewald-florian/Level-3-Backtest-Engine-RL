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


# TODO: Observation space must be increased with some infos which help
#  the agent when he placed his last order and whether it was filled
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
        self.twap_interval = episode_length/twap_n

    def receive_reward(self):

        reward = self.twap_incentive_reward
        return reward


class ActionSpace(BaseActionSpace):
    """Specific Implementation of action space."""
    def __init__(self, verbose=False):
        super().__init__()

        self.verbose = verbose

        self.agent_metrics = AgentMetrics()
        self.market_features = MarketFeatures()

    def take_action(self, action):
        """
        Take action as input and translate into trading decision.
        :param action,
            ..., next action
        """

        # Set done-flag if inventory is completely sold.
        self.check_inventory_sold()

        # submit marketable limit orders
        best_ask = self.market_features.best_ask()
        best_bid = self.market_features.best_bid()
        ticksize = Market.instances["ID"].ticksize

        # define several price limits
        buy_market = best_bid - 100*ticksize  # defacto market-order
        buy_limit_1 = best_ask
        buy_limit_2 = best_ask + ticksize
        buy_limit_3 = best_ask + 2*ticksize

        # -- Zero ending inventory constraint.
        self.zero_ending_inventory_constraint()

        # define several quantities (ratios of initial inv)
        # 5% of initial env
        qt_1 = AgentContext.initial_inventory * 0.05
        # 10% of initial env
        qt_2 = AgentContext.initial_inventory * 0.10
        # 20% of initial env
        qt_3 = AgentContext.initial_inventory * 0.20

        # Note: For now, I have 4 limits and 3 quantities plus "wait" option.
        # -> I need 3*4 + 1 = 13 actions

        order_limit = None
        order_quantity = None

        if action == 0:
            pass

        elif action == 1:
            order_limit = buy_market
            order_quantity = qt_1

        elif action == 2:
            order_limit = buy_market
            order_quantity = qt_2

        elif action == 3:
            order_limit = buy_market
            order_quantity = qt_3

        elif action == 4:
            order_limit = buy_limit_1
            order_quantity = qt_1

        elif action == 5:
            order_limit = buy_limit_1
            order_quantity = qt_2

        elif action == 6:
            order_limit = buy_limit_1
            order_quantity = qt_3

        elif action == 7:
            order_limit = buy_limit_2
            order_quantity = qt_1

        elif action == 8:
            order_limit = buy_limit_2
            order_quantity = qt_2

        elif action == 9:
            order_limit = buy_limit_2
            order_quantity = qt_3

        elif action == 10:
            order_limit = buy_limit_3
            order_quantity = qt_1

        elif action == 11:
            order_limit = buy_limit_3
            order_quantity = qt_2

        elif action == 12:
            order_limit = buy_limit_3
            order_quantity = qt_3

        # Place order via market interface.
        if action > 0:
            self.market_interface.submit_order(side=2,
                                               limit=order_limit,
                                               quantity=order_quantity)
            if self.verbose:
                pass
                #print(f'(RL AGENT) Submission: limit: {order_limit}  qt: {order_quantity}')


class TwapIncentiveAgent(RlBaseAgent):
    """
    Agent with a larger action space, e.g. selection between different
    limits and quantities.
    """
    def __init__(self,
                 initial_inventory: int = 1000_0000_0000,
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
        print("INITIAL INV", AgentContext.initial_inventory)

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

    # TODO: Abstract method brauche ich eigenlich nicht mehr (kann ich in
    #  BaseAgent löschen aber dann werden alte agents failen)
    def _take_action(self, action):
        self.action_space.take_action(action)

    def reset(self):
        super().__init__()
        self.quantity = self.quantity
        self.verbose = self.verbose

        # compositions
        self.market_interface.reset()
        self.reward.reset()
        self.observation_space.reset()
        self.market_features.reset()


