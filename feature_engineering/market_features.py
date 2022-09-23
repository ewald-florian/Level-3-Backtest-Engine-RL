#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 18/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Feature Engineering clas for the Level-3 backtest engine"""


# ---------------------------------------------------------------------------
from context.context import Context
from market.market_trade import MarketTrade

import numpy as np
import pandas as pd

# Compute features based on Context (e.g. midpoint timeseries...)
# Compute features based on MarketTrade (e.g. trading volume)

#TODO: should I define a feature-method and a feature-timeseries method for
# each feature or is there a better way to include timeseries?

#TODO: Wie würde ich es am besten umsetzen wenn ich eine liste mit den letzten
# 1000 midpoints brauche? bzw. mit den letzen 1000 level2_plus arrays?

#TODO: Note: Erstmal nur die features berechnen die ich wirklich brauche
# da ich alles neu machen muss wenn sich zB die Datenstruktur von context
# änder oÄ.!


class MarketFeatures:

    def __init__(self):

        # context container
        #self.context = Context.context_list
        self.context = Context.context_list
        self.market_trade = MarketTrade.history

    # based on market (context) . . . . . . . . . . . . . . . . . . . . . . . .

    def level_2_plus(self, store_timestamp: bool = True,
                     store_hhi: bool = True,
                     data_structure: str = 'array'):
        """
        Data representation of market state which contains level 2 data
        plus additional information based on level 3 data but not complete
        level 3 data.

        Additional feature: concentration index, how strongly concentrated
        the liquidity is on a certain price level (e.g. HHI).

        :param store_timestamp
            bool, if True, ts will be stored

        :param store_hhi
            bool, if True, HHI will be stored

        :param data_structure
            str, 'array' or 'df'
        """

        # check if context is not empty
        if self.context:

            permitted = ['array', 'df']
            if data_structure not in ['array', 'df']:
                raise ValueError(
                    "data_structure must be one of %r." % permitted)

            state_l3 = self.context[-1]
            # store datapoints of current state to a list
            data_list = []

            if store_timestamp:
                data_list.append(state_l3[0])

            for side in [1, 2]:

                for price in state_l3[side].keys():

                    agg_qt = sum([(d['quantity']) for d in state_l3[side][price]])

                    if store_hhi:
                        # herfindahl index
                        hhi = sum([(d['quantity'] / agg_qt) ** 2 for d in
                                   state_l3[side][price]])
                        data_list.extend((price, agg_qt, hhi))
                    else:
                        data_list.extend((price, agg_qt))

            # numpy array
            state_array = np.array(data_list)
            if data_structure == "array":
                return state_array

            # dataframe:
            elif data_structure == 'df':

                state_df = self._convert_to_df(state_array=state_array,
                                    num_levels=len(state_l3[1]),
                                    store_timestamp=store_timestamp,
                                    store_hhi=store_hhi)

                return state_df

            else:
                pass


    def level_2(self, store_timestamp: bool = True,
                data_structure: str = 'df'):
        """
        Level 2 representation of the current state as array or
        dataframe. Note, it could be more convenient to store
        state_l2 directly in context when it is used heavily.
        :param store_timestamp,
            bool, True if ts should be stored in level_2
        :param data_structure
            str, 'array' or 'df', structure of l2
        """
        # get level 2 from level 2 plus method
        level_2 = self.level_2_plus(store_timestamp=store_timestamp,
                                    store_hhi=False,
                                    data_structure=data_structure)

        return level_2

    @staticmethod
    def _convert_to_df(state_array,
                       num_levels:int,
                        store_timestamp: bool = True,
                        store_hhi: bool = True):

        # generate column names
        column_names = []
        if store_timestamp:
            column_names.append('time')

        for side in ['bid', 'ask']:
            for x in range(1, num_levels + 1):
                if store_hhi:
                    column_names.extend((f'l{x}-{side}-price',
                                         f'l{x}-{side}-qt',
                                         f'l{x}-{side}-hhi'))
                else:
                    column_names.extend(
                        (f'l{x}-{side}-price', f'l{x}-{side}-qt'))

        state_df = pd.DataFrame(columns=column_names)
        state_df.loc[0] = state_array

        return state_df

    # TODO
    def weighted_priority_time(self):
        # for each level
        # ptiority_time*qt / sum(qt)
        # test if good signal for midpoint
        pass

    def midpoint(self):
        """
        Current Market midpoint.
        """

        # check if context is not empty
        if self.context:

            best_bid = max(self.context[-1][1].keys())
            best_ask = min(self.context[-1][2].keys())

            return int((best_bid + best_ask)/2)

    def midpoint_series(self):
        pass

    def midpoint_moving_avg(self):
        pass

    def midpoint_moving_avg_series(self):
        pass

    def midpoint_moving_std(self):
        pass

    def midpoint_moving_std_series(self):
        pass

    def _moving_avg(self, window, input_list):
        # helper: compute rolling mean of any input list
        pass

    def _moving_std(self, window, input_list):
        # helper: compute rolling var of any input list
        pass

    def midpoint_expanding_mean(self):
        pass

    def midpoint_expanding_std(self):
        pass

    # TODO: see my old context file (Github)
    def _expanding_mean(self):
        pass

    def _expanding_std(self):
        pass

    def best_bid(self):
        """
        Current best bid.
        """
        # check if context is not empty
        if self.context:
            best_bid = max(self.context[-1][1].keys())
            return best_bid

    def best_bid_series(self):
        pass

    def best_ask(self):
        """
        Current best ask.
        """
        # check if context is not empty
        if self.context:
            best_ask = min(self.context[-1][2].keys())
            return best_ask

    def best_ask_series(self):
        pass

    def rel_spread(self):
        """
        Current relative spread.
        """
        # check if context is not empty
        if self.context:
            best_bid = self.best_bid()
            best_ask = self.best_ask()
            midpoint = self.midpoint()
            rel_spread = (best_ask - best_bid) / midpoint

            return rel_spread

    def rel_spread_series(self):
        pass

    def lob_imbalance(self, num_levels:int=1):
        """
        Current LOB imbalance for a selected number of price levels.
        :param num_levels:
            int, number of price levels to be considered (default 1)
        :return: imbalance
            float, lob imbalance
        """
        # check if context is not empty
        if self.context:

            state_l3 = self.context[-1]

            # aggregate the quantities for the respective number of levels
            bid_keys = list(state_l3[1].keys())[:num_levels]
            bid_qt = sum(
                [sum([d['quantity'] for d in state_l3[1][n]]) for n in bid_keys])

            ask_keys = list(state_l3[2].keys())[:num_levels]
            ask_qt = sum(
                [sum([d['quantity'] for d in state_l3[2][n]]) for n in ask_keys])

            imbalance = (bid_qt - ask_qt) / (ask_qt + bid_qt)

            return round(imbalance, 3)

    def xlm(self):
        pass

    def xlm_series(self):
        pass

    def ohlc(self, window, volume=True):
        pass

    def macd(self):
        pass

    def rsi(self):
        pass

    def bollinger_bands(self):
        pass

    def intraday_season(self):
        # based on clocktime
        # divide trading days in several seasons based on trading volume
        # ie.e is = 1 -> high volume times, ie = 0 -> low volume types
        pass

    # based on market trades . . . . . . . . . . . . . . . . . . . . . . . . .

    def time_since_last_market_trade(self):
        """
        Time since last market trade in nanoseconds.
        :param time_elapsed
            int, time since last trade
        """
        if self.market_trade and self.context[-1][0]:
            time_last_trade = self.market_trade[-1]['execution_time']
            current_time = self.context[-1][0]
            time_elapsed = current_time - time_last_trade

            return time_elapsed

    def cumulative_trading_volume(self):
        """
        Cumulative trading volume since beginning of episode.
        """
        if self.market_trade:
            cumulative_volume = sum(d['quantity'] for d in self.market_trade)
            return cumulative_volume

    def rolling_market_volume(self, window):
        pass

    def market_vwap(self):
        """
        Curren market vwap based on market trades since beginning of episode.
        :return market_vwap
            ...
        """
        if self.market_trade:
            market_vwap = (sum(d['price'] * d['quantity']
                        for d in self.market_trade) /
                    sum(d['quantity'] for
                        d in self.market_trade)) * 1e-8
            return market_vwap

    #TODO: is there any interestin feature which could be used to capture
    # market orders which are matched via execution summary? -> Could be
    # difficult for the RL agent to be affected by these orders while not
    # being able to observe anything from them (they never reach the
    # orderbook). It is in general an important discussion that the agent
    # is confronted with a "partially observable" environment.




