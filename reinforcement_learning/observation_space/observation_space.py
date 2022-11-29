#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 09/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Observation Space for RL-Agent when not using the abstract classes stucture.
"""
# ---------------------------------------------------------------------------
import numpy as np

from feature_engineering.market_features import MarketFeatures
from feature_engineering.agent_features import AgentFeatures

# Note: min max prices are stored for normalization in RL
from reinforcement_learning.observation_space.minmaxprices \
    import MinMaxPriceStorage


class ObservationSpace:

    # class attribute
    observation = None

    # start_date to compute latest min max prices
    def __init__(self):

        # -- static attributes

        #self.min_price = 1_00000000
        #self.max_price = 200_00000000
        self.min_price = MinMaxPriceStorage.min_price
        self.max_price = MinMaxPriceStorage.max_price

        self.min_qt = 1_0000
        self.max_qt = 10000_0000
        self.ticksize = 0.1

        self.market_features = MarketFeatures()
        self.agent_features = AgentFeatures()

        # -- update class attribute
        self.__class__.instance = self.holistic_observation()

    # TODO: implement
    def _get_min_max_prices(self):
        # avoid look-ahead bias...
        # compute min max of the last episode
        # multiply with some margin multiplier or so
        self.min_price = ...
        self.max_price = ...

    def _get_min_max_quantities(self):
        # I don't necessarily need historical values here,
        # can assume min_quantity = 1 and max_quantity as a
        # cap value (e.g. 10_000, depending on symbol price
        # and liquidity...),
        # TODO: maybe I can use the standard market size or normal market size
        #  values defined by xetra.
        # larger quantities will be capped to this value
        self.min_qt = 1
        self.max_qt = ...

    def market_observation(self):
        """
        Create market observation. This usually includes to take lob data
        and additional features from MarketFeatures and normalize them.
        """

        # -- market features
        market_obs = self.market_features.level_2_plus(store_timestamp=False,
                                                  data_structure='array')

        # TODO: added this to avoid some import errors
        if market_obs is not None:
            prices = market_obs[::3]
            quantities = market_obs[1::3]
            # -- normalize
            prices = self._min_max_norma_prices(prices)
            quantities = self._min_max_norma_quantities(quantities)
            market_obs[::3] = prices
            market_obs[1::3] = quantities

        return market_obs

    def _min_max_norma_prices(self, input_array):
        """
        Scales values between 0 and 1.
        """
        scaled_prices = (input_array - self.min_price)/(
                    self.max_price - self.min_price)

        return scaled_prices

    def _min_max_norma_quantities(self, input_array):
        """
        Scales values between 0 and 1.
        """
        scaled_quantities = (input_array - self.min_qt)/(
                    self.max_qt - self.min_qt)

        return scaled_quantities

    def _arbitrary_min_max_normalization(self, input_array, a:int = -1,
                                         b: int = 1):
        """
        Scales values between arbitrary numbers a and b.
        """
        pass

    def _z_score_normalization(self):
        pass

    def agent_observation(self):
        # based on AgentMetrics
        return np.array([])

    def holistic_observation(self):
        """
        Combine market_obs and agent_obs to one array which
        can be fed into the NN.
        """
        market_obs = self.market_observation()
        agent_obs = self.agent_observation()

        holistic_obs = np.append(market_obs, agent_obs)
        holistic_obs.astype('float32')

        return holistic_obs

    def reset(self):
        pass
