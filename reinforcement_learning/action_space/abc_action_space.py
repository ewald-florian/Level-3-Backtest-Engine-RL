#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Abstract Action Space class for RL-Agent
"""
# ----------------------------------------------------------------------------
__author__ = 'florian'
__date__ = '12-20-2022'
__version__ = '0.1'

# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

from market.market_interface import MarketInterface
from reinforcement_learning.transition.env_transition import \
    EnvironmentTransition
from agent.agent_metrics import AgentMetrics
from market.market import Market
from context.agent_context import AgentContext
from feature_engineering.market_features import MarketFeatures


class BaseActionSpace(ABC):
    """
    BaseActionSpace is an abstract class to be subclassed by a specific
    action space. The abstract method take_action must be implemented in the
    subclass. Standard action methods can be stored in the BaseActionSpace.
    """

    # start_date to compute latest min max prices
    def __init__(self):
        """
        Usually, BaseActionSpace is initialized via super in
        the respective ActionSpace. Initialization builds a composition
        of MarketInterface.
        """
        self.market_interface = MarketInterface()
        self.agent_metrics = AgentMetrics()
        self.market_features = MarketFeatures()

        self.final_market_order_submitted = False

    @abstractmethod
    def take_action(self, action):
        """
        Takes action as input and translates it into submissions or
        cancellations via market_interface.
        """
        raise NotImplementedError("Implement market_observation in subclass.")

    def check_inventory_sold(self):
        """
        Set done flag if remaining inventory is zero and hence all inventory
        is sold.
        """
        if self.agent_metrics.remaining_inventory <= 0:
            # Set the done flag to True in the environment transition storage.
            EnvironmentTransition(done=True, info={})

    def zero_ending_inventory_constraint(self,
                                         end_buffer=1e+9,
                                         agent_side=2):
        """
        Checks how much time is left and executes the remaining inventory
        when a given buffer before episode end is reached, e.g. 1 second.
        Note: In order to set the done-flag after the entire inventory is
        sold, use check_inventory_sold()
        :param end_buffer
            int, time buffer before end in nanoseconds.
        """
        current_time = Market.instances["ID"].timestamp
        episode_end = AgentContext.end_time
        time_till_end = episode_end - current_time

        if (time_till_end < end_buffer and
                self.agent_metrics.remaining_inventory > 0
                and not self.final_market_order_submitted):

            # For market buy order: double the best ask.
            if agent_side == 1:
                order_limit = self.market_features.best_ask() * 2
            # For market sell order: half of the best bid.
            else:
                order_limit = self.market_features.best_bid() - \
                              200*Market.instances["ID"].ticksize

            order_quantity = self.agent_metrics.remaining_inventory
            self.market_interface.submit_order(side=2,
                                               limit=order_limit,
                                               quantity=order_quantity)
            # Set flag True to avoid placing the order twice.
            self.final_market_order_submitted = True

    def reset(self):
        """
        Note: the ActionSpace must be composed freshly in agent every
        time the agent gets resets, this resets the ActionSpace
        and the compositions inside ActionSpace automatically.
        """
        pass
