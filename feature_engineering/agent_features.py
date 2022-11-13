#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 18/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Feature Engineering clas for the Level-3 backtest engine"""


# ---------------------------------------------------------------------------

from market.market import Market
from agent.agent_order import OrderManagementSystem
from agent.agent_trade import AgentTrade
from context.agent_context import AgentContext
# TODO:
#  MarketFeatures greift auf Context und auf MarketTrade zurück
#  AgentFeatures könnte auf Agent, AgentTrade, OMS zurückgreifen
#  MarketFeatures braucht keinerlei Klassenvariablen, alles wird on the fly
#  berechent, hier könnte ich schon einige Variablen gebrauchen:
#  start_time --> replay.episode
#  episode_length --> replay.episode
#  end_time (kann ich selbst berechnen) --> replay.episode
#  initial_inventory --> replay.rl_agent
#  ==> Ich muss AgentMetrics in Replay nach Episode und Agent resetten
#  und mir diese variablen von replay.episode und von Agent holen
#  Remaining Inventory:
#  Ich brauche Initial Inventory als static attribute
#  Die Trades und so könnte ich mir on the fly holen
#  Elapsed / Remaining Time:
#  Ich muss den Ersten timestamp abfangen, den rest kann ich mir direkt von
#  Market.timestamp holen
#

# TODO: AgentFeatures can just a a composition inside the agent!
#  but how do I get the timstamps there? I could use the first iteration trick
#  außerdem muss ja eigentlich ObservationSpace auf AgentFeatures zugreifen...

# TODO: Vielleicht macht es mehr Sinn, die AgentFeatures Klasse
#  weg zu lassen und das direkt in agent zu implementieren...? Es ist
#  eigentlich auch relativ Strategiespezifisch / nur für OE zugeschnitten

# TODO: Eine Möglichkeit wäre, initial inventory einfach als class
#  attribut in Agent zu speichern, dann kann AgentTrade darauf zugreifen


class AgentFeatures:

    def __init__(self):

        self.number_of_trades = 0
        self.executed_volume = 0

    @property
    def remaining_inventory(self, normalize=True):
        """Remaining inventory."""
        if AgentContext.initial_inventory:
            initial_inventory = AgentContext.initial_inventory
            # Only update executed_volume when new trades happened
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
                self.executed_volume = sum_quantity

            remaining_inventory = initial_inventory - self.executed_volume

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
            current_time = Market.instances['ID'].timestamp
            start_time = AgentContext.start_time
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

    def time_since_last_submission(self):
        pass

    def number_of_submissions(self):
        pass

    def time_since_last_trade(self):
        pass

    def number_of_trades(self):
        pass

    def inventory(self): # postion
        pass

    def reset(self):
        pass