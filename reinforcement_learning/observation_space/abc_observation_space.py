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
from reinforcement_learning.observation_space.minmaxprices \
    import MinMaxPriceStorage


class BaseObservationSpace(ABC):
    """
    BaseObservationSpace is an abstract method to be subclassed by a specific
    observation space. The abstract classes agent_observation and
    market_observation must be implemented in the subclass.
    """

    # start_date to compute latest min max prices
    def __init__(self):
        # TODO: include if self.min /self.max etc. to avoid error
        #self.min_price = MinMaxPriceStorage.min_price
        #self.max_price = MinMaxPriceStorage.max_price

        #self.min_qt = 1_0000
        #self.max_qt = MinMaxPriceStorage.max_qt

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

    def _min_max_norma_prices(self, input_array):
        """
        Scales values between 0 and 1.
        """
        scaled_prices = (input_array - MinMaxPriceStorage.min_price)/(
            MinMaxPriceStorage.max_price - MinMaxPriceStorage.min_price)

        return scaled_prices

    def _min_max_norma_quantities(self, input_array):
        """
        Scales values between 0 and 1.
        """
        scaled_quantities = (input_array - MinMaxPriceStorage.min_qt)/(
                    MinMaxPriceStorage.max_qt - MinMaxPriceStorage.min_qt)

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

    def reset(self):
        """
        Note: the ObservationSpace must be composed freshly in agent every
        time the agent gets resetted, this resets the ObservationSpace
        and the compositions inside ObservationSpace automatically.
        """
        pass

