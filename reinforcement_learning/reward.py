#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 17/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Reward class for RL Environment.
"""
# ---------------------------------------------------------------------------
from agent.agent_metrics import AgentMetrics
#TODO: just implement each possible reward function as a new method such that
# they can be freely selected by different agents and compared


class Reward:

    # class attribute
    reward = None

    def __init__(self):

        self.agent_metrics = AgentMetrics()
        self.last_pnl = 0

    @property
    def pnl_unrealized(self):

        return self.agent_metrics.pnl_unrealized

    @property
    def pnl_realized(self):

        return self.agent_metrics.pnl_realized

    # TODO: testing
    @property
    def pnl_marginal(self):
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