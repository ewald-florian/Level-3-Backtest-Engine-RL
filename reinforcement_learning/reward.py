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
# they can be freely choosen anc compared

class Reward:

    def __init__(self):
        pass

    def pnl_unrealized(self):

        return AgentMetrics.pnl_unrealized

    def unrealized_pnl(self):

        return AgentMetrics.pnl_realized

    def pnl_difference(self):
        # basically, pnl of the most recent roundtrip?
        # new_pnl_real - old_pnl_real
        pass

    def vwap_score(self):
        pass

    def relative_buy_vwap(self):
        pass

    def relative_sell_vwap(self):

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
        pass