#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : florian
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Observation Space for RL-Agent when not using the abstract classes structure.
"""
# ---------------------------------------------------------------------------
import numpy as np

from feature_engineering.market_features import MarketFeatures
from feature_engineering.agent_features import AgentFeatures

# Note: min max prices are stored for normalization in RL
from reinforcement_learning.observation_space.minmaxvalues \
    import MinMaxValues


class ObservationSpace:

    # class attribute
    observation = None

    # start_date to compute latest min max prices
    def __init__(self):

        # -- static attributes

        #self.min_price = 1_00000000
        #self.max_price = 200_00000000
        self.min_price = MinMaxValues.min_price
        self.max_price = MinMaxValues.max_price

        self.min_qt = 1_0000
        self.max_qt = MinMaxValues.max_qt

        self.market_features = MarketFeatures()
        self.agent_features = AgentFeatures()

        # -- update class attribute
        self.__class__.instance = self.holistic_observation()

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

        # DEBUGGING
        print("MAX QT", self.max_qt)

        return holistic_obs

    def reset(self):
        pass
