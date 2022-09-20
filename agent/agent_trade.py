#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Agent Trade storage class for the Level-3 backtest engine"""

# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np


class AgentTrade: # vs MarketTrade

    history = list()

    def __init__(self,
                 agent_trade: dict,
                 verbose: bool = False):
        """
        Store agent trades in class attribute history (list)
        :param agent_trade
            dict, trade information
        :param verbose
            bool, set True to log trades
        """
        # append trade dict to history list
        self.__class__.history.append(agent_trade)

        trade_id = agent_trade['trade_id']
        execution_time = agent_trade['execution_time']
        executed_volume = agent_trade['executed_volume']
        execution_price = agent_trade['execution_price']
        #self.aggressor_side = agent_trade['aggressor_side']
        #self.message_id = agent_trade['message_id']
        agent_side = agent_trade['agent_side']

        #TODO: Format f string
        if verbose:
            print(f'(EXEC-INFO)agent-trade {trade_id} executed|{executed_volume}@{execution_price}|side: {agent_side}|time: {pd.to_datetime(int(execution_time), unit="ns")}|unix {execution_time}')

    @property
    def dataframe(self):
        """
        Dataframe representation of trade history.
        """
        return pd.DataFrame.from_records(
            self.__class__.history)

    @property
    def array(self):
        """
        Numpy Array representation of trade history.
        """
        df = self.dataframe
        return np.array(df)

    @property
    def trade_count(self):
        """
        Trade count
        """
        return len(self.__class__.history)

    #TODO: implement helpful properties...
    @property
    def vwap_buy(self):
        pass

    @property
    def vwap_sell(self):
        pass

    @classmethod
    def reset_history(cls):
        """
        Reset trade history.
        """
        # delete all elements in Trade.history (list)
        del cls.history[:]