#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 09/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Observation Space for RL-Agent
"""
# ---------------------------------------------------------------------------

from feature_engineering.market_features import MarketFeatures
from feature_engineering.agent_features import AgentFeatures

import numpy as np
# TODO: Überlegen wie und wo ObservationSpace aufgerufen werden soll
# TODO: Welche art von class attribute (einfach self? -> self.observation)

#TODO: Concept: get "raw" features from FeatureEngineering, normalize and
# concatenate them in ObservationSpace, pass to Neural Network (obs)


class ObservationSpace:

    # class attribute
    observation = None

    # start_date to compute latest min max prices
    def __init__(self, start_date=None):

        # -- static attributes
        self.min_price = 258.4
        self.max_price = 302.4
        self.min_size = 1.0
        self.max_size = 50614.0
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
        # and liquidity...)
        # larger quantities will be capped to this value
        self.min_qt = 1
        self.max_qt = ...

    def market_observation(self):
        # -- market features
        market_obs = self.market_features.level_2_plus()

        # -- normalize


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
        pass

    def holistic_observation(self):
        """
        Combine market_obs and agent_obs to one array which
        can be fed into the NN.
        """
        market_obs = self.market_observation
        agent_obs = self.agent_observation

        holistic_obs = np.append(market_obs, agent_obs)
        #holistic_obs.astype('float32')

        return holistic_obs

    def reset(self):
        pass
