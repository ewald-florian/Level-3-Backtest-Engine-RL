#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Abstract Observation Space class for RL-Agent
"""
#----------------------------------------------------------------------------
__author__ =  'florian'
__version__ = '0.1'
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np

from feature_engineering.market_features import MarketFeatures
from feature_engineering.agent_features import AgentFeatures
from reinforcement_learning.observation_space.minmaxvalues \
    import MinMaxValues
from agent.agent_order import OrderManagementSystem as OMS
from context.agent_context import AgentContext


class BaseObservationSpace(ABC):
    """
    BaseObservationSpace is an abstract class to be subclassed by a specific
    observation space. The abstract methods agent_observation and
    market_observation must be implemented in the specific class.
    """

    # start_date to compute latest min max prices
    def __init__(self):
        """
        Usually, BaseObservationSpace is initialized via super in
        the respective ObservationSpace. Initialization builds compositions
        of MarketFeatures and AgentFeatures.
        """
        self.market_features = MarketFeatures()
        self.agent_features = AgentFeatures()

        self.number_of_orders = 0

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

    def _min_max_norma_prices_clipped(self, input_array):
        """
        Scales values between 0 and 1 with clipping of outliers to max.
        """

        scaled_prices = (input_array - MinMaxValues.min_price) / (
                MinMaxValues.max_price - MinMaxValues.min_price)

        # Clip outliers
        scaled_prices[scaled_prices > 1] = 1
        scaled_prices[scaled_prices < 0] = 0

        return scaled_prices

    def _min_max_norma_quantities_clipped(self, input_array):
        """
        Scales values between 0 and 1 with clipping of outliers to max.
        """
        scaled_quantities = (input_array - MinMaxValues.min_qt) / (
                MinMaxValues.max_qt - MinMaxValues.min_qt)

        # Clip outliers
        scaled_quantities[scaled_quantities > 1] = 1
        scaled_quantities[scaled_quantities < 0] = 0

        return scaled_quantities

    def _min_max_norma_prices(self, input_array):
        """
        Scales values between 0 and 1.
        """
        scaled_prices = (input_array - MinMaxValues.min_price) / (
                MinMaxValues.max_price - MinMaxValues.min_price)

        return scaled_prices

    def _min_max_norma_quantities(self, input_array):
        """
        Scales values between 0 and 1.
        """
        scaled_quantities = (input_array - MinMaxValues.min_qt) / (
                MinMaxValues.max_qt - MinMaxValues.min_qt)

        return scaled_quantities

    def _arbitrary_min_max_normalization(self, input_array, a: int = -1,
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

    @property
    def standard_agent_observation(self) -> np.array:
        """
        This is the standard agent observation it consist of two values:
        1. elapsed time
        2. elapsed inventory.
        """
        # Note: very common agent state, see e.g. Beling/Liu
        # TODO: Habe ich hier norm vergessen???
        time = self.agent_features.elapsed_time
        inv = self.agent_features.remaining_inventory
        time_since_last_sub = \
            self.agent_features.time_since_last_submission_norm
        agent_obs = np.array([time, inv, time_since_last_sub])
        return agent_obs

    def reset(self):
        """
        Note: the ObservationSpace must be composed freshly in agent every
        time the agent gets resetted, this resets the ObservationSpace
        and the compositions inside ObservationSpace automatically.
        """
        pass

