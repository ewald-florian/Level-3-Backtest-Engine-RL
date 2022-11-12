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


class AgentFeatures:

    def __init__(self):

        # In UNIX
        self.start_time = None
        # In nanoseconds
        self.episode_length = None
        # end_time = start_time + episode_length
        self.end_time = None

    def remaining_inventory(self):
        pass

    def elapsed_time(self, normalize=True):
        """Elapsed Time of the current Episode.
        :param normalize
            bool, normalized between 0 and 1 if set to True
        """
        current_time = Market.instances['ID'].timestamp
        elapsed_time = (current_time-self.start_time)
        if normalize:
            elapsed_time = elapsed_time / self.episode_length
        return elapsed_time

    def remaining_time(self, normalize=True):
        current_time = Market.instances['ID'].timestamp
        remaining_time = (self.end_time - current_time)
        if normalize:
            remaining_time = remaining_time / self.episode_length
        return remaining_time

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