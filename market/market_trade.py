#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to store Market Trades (from execution summary messages)
"""


import pandas as pd
import numpy as np


class MarketTrade:

    history = list()

    def __init__(self,
                 market_trade: dict,
                 verbose: bool = False):
        """
        Store historical market trades in class attribute history (list)
        :param market_trade
            dict, trade information
        :param verbose
            bool, set True to log trades
        """

        # append trade dict to history list
        self.__class__.history.append(market_trade)

        if verbose:
            print(f'(Market-Execution)')

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
        return self.__class__.history[-1]['market_trade_num']

    @classmethod
    def reset_history(cls):
        """
        Reset trade history.
        """
        # delete all elements in Trade.history (list)
        del cls.history[:]

    def __str__(self):
        pass

