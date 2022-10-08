#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Abstract Observation Space class for RL-Agent
"""
#----------------------------------------------------------------------------
__author__ =  'florian'
__date__ =  '08-10-2022'
__version__ = '0.1'
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np

from feature_engineering.market_features import MarketFeatures
from feature_engineering.agent_features import AgentFeatures
# TODO: Überlegen wie und wo ObservationSpace aufgerufen werden soll
# TODO: Welche art von class attribute (einfach self? -> self.observation)
# TODO: Eigentlich unpraktisch da ich zuerst ObservationSpace() callen müsste
#  um die aktuelle observation zu bekommen und erst dann
#  ObservationSpace.observation bekommen könnte...
#  Evtl. ist es besser ObservationSpace.holistic_observation einfach direkt zu
#  callen...


#TODO: Concept: get "raw" features from FeatureEngineering, normalize and
# concatenate them in ObservationSpace, pass to Neural Network (obs)


class BaseObservationSpace(ABC):
    """
    BaseObservationSpace is an abstract method to be subclassed by a specific
    observation space. The abstract classes agent_observation and
    market_observation must be implemented in the subclass.
    """

    # start_date to compute latest min max prices
    def __init__(self):

        # -- static attributes
        # TODO: remove hardcode
        self.min_price = 1_00000000
        self.max_price = 200_00000000
        # TODO: remove hardcode
        self.min_qt = 1_0000
        self.max_qt = 10000_0000
        self.ticksize = 0.1

        self.market_features = MarketFeatures()
        self.agent_features = AgentFeatures()

    @abstractmethod
    def market_observation(self):
        """
        Create market observation. This usually includes to take lob data
        and additional features from MarketFeatures and normalize them.
        """
        raise NotImplementedError("Implement market_observation in subclass.")

    @abstractmethod
    def agent_observation(self):
        """
        Abstract method must be implemented in subclass. Create agent
        observation. Usually based on agent metrics such as exposure, position
        remaining time of the trading period, inventory.
        """
        raise NotImplementedError("Implement agent_observation in subclass.")

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

