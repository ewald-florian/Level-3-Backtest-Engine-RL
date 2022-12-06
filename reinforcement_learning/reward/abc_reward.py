#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Abstract Reward class for RL-Agent
"""
#----------------------------------------------------------------------------
__author__ =  'florian'
__date__ =  '08-10-2022'
__version__ = '0.1'
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

from agent.agent_metrics import AgentMetrics
from agent.agent_trade import AgentTrade
#TODO: just implement each possible reward function as a new method such that
# they can be freely selected by different agents and compared

# TODO: TWAP-Comparisions are difficult for me since I, to some part assume
#  a market impact, I cannot just run TWAP and RL in parallel sicne they would
#  "stel" liquidity of each other... One effortful solution would be to take
#  the exact same episode, run both TWAP and RL separately one after another
#  and then compare performance... could only be used for a sparse reward in
#  the end, would be rather inefficient. Same goes for VWAP

# TODO: for a short trading horizon like 1 min, VWAP is not really interesting
#  as benchmark.


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

    @abstractmethod
    def receive_reward(self):
        """
        Abstract method to be implemented in subclass. This could either be
        a custom reward function or just calling a standard reward function
        as given by the properties below.
        """
        raise NotImplementedError("Implement receive_reward in subclass.")

    @property
    def last_trade_is(self):
        """Returns 0 if no trade happened and IS of last trade if a new
        trade happened. """
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
        return latest_trade_is

    def episode_end_is(self, last_episode_step):
        """IS over all trades at the end of the episode. This method returns
        the overall is over all trades, in order to be used as terminal reward
        it needs the input argument last_episode_step which is a boolean
        that should only be True if the environment is in the last episode.
        :param last_episode_step
            bool, True if last_episode_step False otherwise.
        """
        episode_end_is = 0
        if last_episode_step:
            episode_end_is = self.agent_metrics.all_trade_is()
        return episode_end_is



    @property
    def pnl_unrealized(self):

        return self.agent_metrics.pnl_unrealized

    @property
    def pnl_realized(self):

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
        self.last_pnl = 0
        self.agent_metrics = AgentMetrics()