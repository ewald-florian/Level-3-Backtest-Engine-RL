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
from context.context import Context


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

    def zero_ending_inventory_constraint(self, agent_side=2):
        """
        Submit remaining inv as market order in the second last step.
        :param done_flag
            bool, Episode done
        :param agent_side
            int, side of the trading agent (e.g. 2 for liquidation agent)
        """
        if EnvironmentTransition.transition[0] and \
                self.agent_metrics.remaining_inventory > 0:

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
        else:
            pass

    def zero_ending_inventory_constraint_old(self,
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

        #DEBUGGING
        print("zero-ending")
        print(current_time)
        print(episode_end)
        print(time_till_end - end_buffer)
        print("reminv", self.agent_metrics.remaining_inventory)
        print("flag", self.final_market_order_submitted)


        if (time_till_end < end_buffer and
                self.agent_metrics.remaining_inventory > 0
        # TODO: die flag ist immer True ab der 2. Episode, reset richten???
                and not self.final_market_order_submitted):

            # DEBUGGING
            print("zero-ending-inv is triggered!")

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

    def limit_and_qt_action(self, action):
        """
        Execution of a discrete action for OE which determines both
        limit price and quantity via MarketInterface.
        """
        # -- Set done-flag if inventory is completely sold.
        self.check_inventory_sold()
        # -- Zero ending inventory constraint.
        self.zero_ending_inventory_constraint()

        # DEBUGGING MODE
        #action = 0
        ###############

        # Base variables for limit and qt.
        # I use Context since it contains the (blocked) simulation state.
        best_bid = list(Context.context_list[-1][1].keys())[0]
        best_bid_qt_list = Context.context_list[-1][1][best_bid]
        best_bid_qt = sum([d['quantity'] for d in best_bid_qt_list])

        best_ask = list(Context.context_list[-1][2].keys())[0]
        second_best_ask = list(Context.context_list[-1][2].keys())[1]
        third_best_ask = list(Context.context_list[-1][2].keys())[2]
        ticksize = Market.instances["ID"].ticksize

        # DEBUGGING
        print("Context")
        print(Context.context_list[-1][1].keys())
        print(Context.context_list[-1][2].keys())
        print(best_bid)
        print(best_bid_qt_list)
        print(best_bid_qt)
        print(best_ask)
        print(second_best_ask)
        print(third_best_ask)

        # -- Define several price limits.
        market_order = best_bid - 100 * ticksize  # defacto market-order
        buy_limit_1 = best_ask
        buy_limit_2 = second_best_ask
        buy_limit_3 = third_best_ask

        # TODO 1: Welche Qts machen am meisten Sinn?
        # -- define several quantities (ratios of initial inv).
        #  5% of initial env
        qt_1 = int(AgentContext.initial_inventory * 0.05)
        # 10% of initial env
        qt_2 = int(AgentContext.initial_inventory * 0.10)
        # 20% of initial env
        qt_3 = int(AgentContext.initial_inventory * 0.20)
        # TWAP Quantity (based on provided number of twap intervals)
        # TODO: num twap kann ich in AgentContext speichern.
        twap_qt = AgentContext.initial_inventory / self.num_twap_intervals

        # Note: For now, I have 4 limits and 3 quantities plus "wait" option.
        # -> I need 3*4 + 1 = 13 actions

        # -- Execute Actions.
        order_limit = None
        order_quantity = None

        # Wait and do nothing during this step.
        if action == 0:
            pass

        # Markatable limit order which targets the quantity on the
        # best bid level.
        elif action == 1:
            order_limit = best_bid
            order_quantity = best_bid_qt

        # TWAP Qt at market.
        elif action == 2:
            order_limit = market_order
            order_quantity = twap_qt

        # Combinations for buy_limit_1
        elif action == 3:
            order_limit = buy_limit_1
            order_quantity = qt_1

        elif action == 4:
            order_limit = buy_limit_1
            order_quantity = qt_2

        elif action == 5:
            order_limit = buy_limit_1
            order_quantity = qt_3

        # Combinations for buy_limit_2
        elif action == 6:
            order_limit = buy_limit_2
            order_quantity = qt_1

        elif action == 7:
            order_limit = buy_limit_2
            order_quantity = qt_2

        elif action == 8:
            order_limit = buy_limit_2
            order_quantity = qt_3

        # Combinations for buy_limit_3
        elif action == 9:
            order_limit = buy_limit_3
            order_quantity = qt_1

        elif action == 10:
            order_limit = buy_limit_3
            order_quantity = qt_2

        elif action == 11:
            order_limit = buy_limit_3
            order_quantity = qt_3

        # Place order via market interface.
        if action > 0:
            self.market_interface.submit_order(side=2,
                                               limit=order_limit,
                                               quantity=order_quantity)

            print(f'(RL AGENT) Submission: limit: {order_limit}  '
                  f'qt: {order_quantity}'
                  f'action: {action}')

    # TODO: Just use the qts of  limit_and_qt_action() when finalised.
    def qt_action(self, action):
        """
        Execution of a discrete action for OE which determines only the
        quantity which is then submitted via market order.
        """
        # -- Set done-flag if inventory is completely sold.
        self.check_inventory_sold()
        # -- Zero ending inventory constraint.
        self.zero_ending_inventory_constraint()

    def limit_price_action(self, action):
        """
        Execution of a discrete action for OE which determines only the
        limit price of an order which is then submitted with a given
        quantity.
        """
        pass

    def reset(self):
        """
        Note: the ActionSpace must be composed freshly in agent every
        time the agent gets resets, this resets the ActionSpace
        and the compositions inside ActionSpace automatically.
        """
        pass
