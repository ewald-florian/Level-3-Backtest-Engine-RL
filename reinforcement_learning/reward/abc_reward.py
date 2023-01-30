#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Abstract Reward class for RL-Agent
"""
# ----------------------------------------------------------------------------
__author__ = 'florian'
__date__ = '08-10-2022'
__version__ = '0.1'

# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

from agent.agent_metrics import AgentMetrics
from agent.agent_trade import AgentTrade
from agent.agent_order import OrderManagementSystem as OMS
from context.agent_context import AgentContext
from reinforcement_learning.transition.env_transition import \
    EnvironmentTransition
from market.market import Market
from reinforcement_learning.action_space.action_storage import ActionStorage


class BaseReward(ABC):
    """
    Abstract reward class to be subclassed.
    """

    # class attribute
    reward = None

    def __init__(self):
        """To be initialized in specific reward class."""

        self.agent_metrics = AgentMetrics()
        self.last_pnl = 0
        # Agent Trade counter for trade specific rewards
        self.number_of_trades = 0
        self.number_or_orders = 0
        self.recorded_orders = 0

    @abstractmethod
    def receive_reward(self):
        """
        Abstract method to be implemented in subclass. This could either be
        a custom reward function or just calling a standard reward function
        as given by the properties below.
        """
        raise NotImplementedError("Implement receive_reward in subclass.")

    def immediate_absolute_is_reward(self, scaling_factor=10):
        """
        Returns 0 if no trade happened and IS of last trade if new
        trades happened.
        :param scaling_factor:
            flaot, scale the reward.
        """
        # set is to 0
        latest_trade_is = 0
        # Check if new trades occurred.
        num_new_trades = len(AgentTrade.history) - self.number_of_trades
        if num_new_trades:
            # Get volume-weighted latest_trade_is from agent_metrics method.
            latest_trade_is = self.agent_metrics.latest_trade_is(
                number_of_latest_trades=num_new_trades)
            # Update class intern trade counter.
            self.number_of_trades = len(AgentTrade.history)
        # return is as reward
        return latest_trade_is * scaling_factor

    def terminal_absolute_is_reward(self, scaling_factor=1):
        """IS over all trades at the end of the episode. This method returns
        the overall IS over all trades, in order to be used as terminal reward
        it needs the input argument last_episode_step which is a boolean
        that should only be True if the environment is in the last episode.

        The done-flag can be used as last_episode_flag if it is determined
        before the reward is called.
        :param scaling_factor:
            flaot, scale the reward.
        """
        episode_end_is = 0
        # If done, compute cumulative volume weighted IS
        # if done:
        if EnvironmentTransition.transition[0]:
            episode_end_is = self.agent_metrics.overall_is()
        return episode_end_is * scaling_factor

    def incentivize_waiting(self, reward_factor=0.002):
        """
        Return a small reward for waiting to incentivice the
        agent to take this action more often. This reward function can also
        be used for pretraining.
        :param reward_factor
            float, reward factor which the agent should earn for waiting.
        """
        if ActionStorage.action == 0:
            return reward_factor
        else:
            return 0

    @property
    def pnl_unrealized(self):
        """Unrealized PnL"""
        return self.agent_metrics.pnl_unrealized

    @property
    def pnl_realized(self):
        """Realized PnL."""
        return self.agent_metrics.pnl_realized

    # TODO: testing
    @property
    def pnl_marginal(self):
        """
        Sparse reward which is only not zero when the pnl realized changes.
        Should approximate the pnl of the latest roundtrip when a roundtrip
        is completed.
        """
        # basically, pnl of the most recent roundtrip?
        # new_pnl_real - old_pnl_real
        # Note only works when called every pnl update such that
        # self.last_pnl is updated
        pnl_difference = self.pnl_realized - self.last_pnl
        # update last pnl
        self.last_pnl = self.pnl_realized
        return pnl_difference

    def twap_time_incentive_reward(self, num_twap_steps=20):
        """
        This is a TWAP based reward which rewards the agent if he
        submits his orders closer to a TWAP-strategy with a given
        number of orders N. This reward is used for pretraining to incentivice
        the agent to better allocate submissions over the entire
        episode and to avoid liquidating everything just in the beginning.

        The reward signal is scaled and clipped in the range [0,5].
        :param num_twap_steps
            int, number of TWAP child orders.
        """
        twap_reward = 0

        # Check if the agent has submitted a new order.
        if len(OMS.order_list) > self.recorded_orders:
            # Special case for first order: The reward is based on the difference
            # between episode start time and first order timestamp. The agent
            # should be incentivized to place his first order as quickly as
            # possible.
            if len(OMS.order_list) == 1:
                current_order_timestamp = OMS.order_list[-1]['timestamp']

                start_time_deviation = -abs(AgentContext.start_time -
                                           current_order_timestamp)

                max = 0
                min = (-AgentContext.episode_length / num_twap_steps)
                y = (start_time_deviation - min) * 5 / (max - min)

                # Clip.
                if twap_reward > 5:
                    twap_reward = 5
                elif twap_reward < 0:
                    twap_reward = 0
                # Count the order.
                self.recorded_orders += 1

            # For the following orders.
            else:
                # Compute the differ between the current and the latest order.
                current_order_timestamp = OMS.order_list[-1]['timestamp']
                last_order_timestamp = OMS.order_list[-2]['timestamp']
                agent_order_difference = abs(last_order_timestamp -
                                             current_order_timestamp)
                # Compute the deviation of the order from the TWAP-interval
                twap_interval_length = (AgentContext.episode_length /
                                        num_twap_steps)

                twap_time_deviation = - abs(twap_interval_length -
                                          agent_order_difference)

                # Scale the reward in the range [0, 5] such that a deviation of
                # 0 yields the highest reward and a deviation of episode
                # len yields the lowest reward.

                max = 0
                min = (-AgentContext.episode_length / num_twap_steps)
                twap_reward = (twap_time_deviation - min)*5 / (max - min)

                # Clip:
                if twap_reward > 5:
                    twap_reward = 5
                elif twap_reward < 0:
                    twap_reward = 0
                # Count the order.
                self.recorded_orders += 1

        # Clip between 0 and 5.
        return twap_reward

    @property
    def twap_incentive_reward_old(self):
        """
        This reward component incentivices the agent to adjust his strategy
        to TWAP strategy. The intuition behind that is that an execution
        agent can learn twap as a baseline and then later spezialize on a
        more sophisticated strategy. The main goal is to limit the number
        of orders and the submission frequency of the agent.
        """
        # Set to zero if no new orders were submitted.
        twap_reward = 0
        # Special case first order: penalty for waiting.
        if len(OMS.order_list) == 1:
            # -- Time aspect.
            current_order_timestamp = OMS.order_list[-1]['timestamp']
            episode_start_time = AgentContext.start_time
            agent_order_time_difference = abs(current_order_timestamp -
                                              episode_start_time)
            twap_time_deviation = abs(self.twap_interval -
                                      agent_order_time_difference)
            # Convert to seconds.
            twap_time_deviation = int(twap_time_deviation / 1e9)
            # -- Quantity aspect.
            current_order_qt = OMS.order_list[-1]['quantity']
            twap_qt_deviation = abs(
                current_order_qt - self.twap_child_quantity)
            # Convert to pieces
            twap_qt_deviation = twap_qt_deviation / 1_0000
            # -- Combination.
            # TODO: must be scaled since ns are too dominant.
            twap_penalty = - twap_time_deviation - twap_qt_deviation
            # Count the new order.
            self.number_of_orders += 1

        # After the first order: compare time between orders.
        num_new_orders = len(OMS.order_list) - self.number_of_orders
        if num_new_orders and len(OMS.order_list) > 2:
            # -- Time aspect.
            current_order_timestamp = OMS.order_list[-1]['timestamp']
            last_order_timestamp = OMS.order_list[-2]['timestamp']
            agent_order_difference = abs(last_order_timestamp -
                                         current_order_timestamp)

            twap_time_deviation = abs(self.twap_interval -
                                      agent_order_difference)
            # Convert to seconds.
            twap_time_deviation = int(twap_time_deviation / 1e9)
            # -- Quantity aspect.
            current_order_qt = OMS.order_list[-1]['quantity']
            twap_qt_deviation = abs(current_order_qt -
                                    self.twap_child_quantity)
            # Convert to pieces
            twap_qt_deviation = twap_qt_deviation / 1_0000

            # -- Combination.
            twap_penalty = - twap_qt_deviation - twap_time_deviation

            # Count the new order.
            self.number_of_orders += 1

        return twap_penalty

    def reward_for_waiting(self):
        """
        Give a positive reward if the agent chooses the action to wait
        and do nothing in the current episode.
        :param action_equals_waiting
            bool, True if wait, False otherwise
        :return wait_reward
            int, reward for waiting
        """
        wait_reward = 0
        if ActionStorage.action == 0:
            wait_reward = 1

        return wait_reward

    def vwap_score(self):
        pass

    def relative_buy_vwap(self):
        pass

    def relative_sell_vwap(self):
        pass

    def timing_reward(self, ideal_trading_interval):
        # Idea: based on the time difference between submissions,
        # reward the agent for achieving a given trading interval
        # so that the trading frequency of the agent can be controlled
        # via the reward. The agent is still free to trade whenever he sees a
        # good opportunity but he will adjust his trading activity towards
        # the (unfortunately arbitrarily chosen) ideal trading interval.
        pass

    def cancellation_reward(self):
        pass

    def reset(self):
        """
        Reset reward class.
        """
        self.__class__.reward = None
        self.recorded_orders = 0
        self.last_pnl = 0
        self.agent_metrics = AgentMetrics()
