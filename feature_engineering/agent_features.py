#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 18/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Feature Engineering clas for the Level-3 backtest engine"""


# ---------------------------------------------------------------------------

import pandas as pd

from market.market import Market
from agent.agent_order import OrderManagementSystem
from agent.agent_trade import AgentTrade
from context.agent_context import AgentContext
from agent.agent_metrics import AgentMetrics


class AgentFeatures:
    """Class to compute agent-features such as remaining time or remaining
    inventory. AgentFeatures is used by ObservationSpace to include the
    agent-state / private-state into the observation."""

    def __init__(self):
        """Initializing sets the number_of_trades and the executed_quantity
        to 0. These are then counted during the episode."""

        self.number_of_trades = 0
        self.executed_quantity = 0
        self.agent_metrics = AgentMetrics()

    @property
    def remaining_inventory(self, normalize=True):
        """Remaining inventory."""
        if AgentContext.initial_inventory:
            initial_inventory = AgentContext.initial_inventory
            # Only update executed_quantity when new trades happened
            num_new_trades = len(AgentTrade.history) - self.number_of_trades
            if num_new_trades:
                # Sum All Trades:
                # TODO: differentiate sides etc. to use the method more general
                sum_quantity = 0
                for trade in AgentTrade.history:
                    quantity = trade['executed_volume']
                    sum_quantity += quantity
                    self.number_of_trades += 1

                # update executed volume
                self.executed_quantity = sum_quantity

            remaining_inventory = initial_inventory - self.executed_quantity

            # Normalize between 0 and 1 by dividing by the initial inventory
            if normalize:
                remaining_inventory = remaining_inventory/initial_inventory

            return remaining_inventory
        # Return remaining_inventory = 1 for the initial_observation.
        else:
            return 1

    @property
    def elapsed_time(self, normalize=True):
        """Elapsed Time of the current Episode.
        :param normalize
            bool, normalized between 0 and 1 if set to True
        """
        if AgentContext.start_time:
            # Market time in unix.
            current_time = Market.instances['ID'].timestamp
            # Start_time in unix.
            start_time = AgentContext.start_time
            # Difference.
            elapsed_time = current_time-start_time
            if normalize:
                episode_length = AgentContext.episode_length
                elapsed_time = elapsed_time / episode_length
            return elapsed_time
        # For the initial observation, elapsed time normed is 0.
        else:
            return 0

    @property
    def remaining_time(self, normalize=True):
        """Remaining Time of the current Episode.
        :param normalize
            bool, normalized between 0 and 1 if set to True
        """
        if AgentContext.end_time:
            current_time = Market.instances['ID'].timestamp
            end_time = AgentContext.end_time
            remaining_time = end_time - current_time
            if normalize:
                episode_length = AgentContext.episode_length
                remaining_time = remaining_time / episode_length
            return remaining_time
        # For the initial observation, normed remaining_time is 1
        else:
            return 1

    @property
    def time_since_last_submission_norm(self):
        """
        Time since last submission normed. For now, it is normed by dividing
        by 30 seconds.
        """
        time_since_last_submission = \
            self.agent_metrics.time_since_last_submission
        # Scale with 30 seconds as max time.
        time_normed = (time_since_last_submission / (30*1e9))
        # Clip to 1 max. Lower clip is 0 by design.
        if time_normed > 1:
            time_normed = 1
        return time_normed

    @property
    def time_since_last_trade_norm(self):
        """
        Time since last trade normed. For now, it is normed by dividing
        by 30 seconds.
        """
        time_since_last_trade = \
            self.agent_metrics.time_since_last_trade
        # Scale with 30 seconds as max time.
        time_normed = (time_since_last_trade / (30 * 1e9))
        # Clip to 1 max. Lower clip is 0 by design.
        if time_normed > 1:
            time_normed = 1
        return time_normed

    def number_of_submissions(self):
        pass

    def time_since_last_trade(self):
        pass

    def number_of_trades(self):
        pass

    def inventory(self): # postion
        pass

    def reset(self):
        """Reset AgentFeatures"""
        self.number_of_trades = 0
        self.executed_quantity = 0