#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#TODO: I could also store agent trades and markt trades in the same class
# in two different lists (market_trades, and market_trades)


class MarketTrade:

    history = list()

    # TODO: save as dict not as np.array()
    def __init__(self, market_trade, verbose=False):
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
        df = self.dataframe()
        return np.array(df)

    @property
    def trade_count(self):
        """
        Trade count
        """
        return self.__class__.history[-1]['market_trade_num']

    #TODO: test / debug vwap / wie kann vwap angezeigt werden wenn es keine instanz gibt?
    @property
    def vwap(self):
        """
        Market vwap of current episode.
        :return
            ...
        """
        vwap = (sum(trade['price']*trade['quantity']
                    for trade in self.__class__.history) /
                    sum(trade['quantity'] for
                    trade in self.__class__.history))*1e-8
        return vwap

    @classmethod
    def reset_history(cls):
        """
        Reset trade history.
        """
        # delete all elements in Trade.history (list)
        del cls.history[:]

