#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract Observation Space class for RL-Agent
"""


from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

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

    @property
    def raw_market_features(self):
        """
        Returns the standard representation of raw market features
        including prices, quantities and hhi in form of 1D np array.
        """
        raw_obs = self.market_features.level_2_plus(store_timestamp=False,
                                                    data_structure='array',
                                                    store_hhi=True)
        if raw_obs is not None:
            prices = raw_obs[::3]
            quantities = raw_obs[1::3]
            # Note: HHI is naturally normed between 0 and 1.
            # hhis = raw_obs[2::3]
            # -- normalize
            prices = self._min_max_norma_prices_clipped(prices)
            quantities = self._min_max_norma_quantities_clipped(quantities)
            raw_obs[::3] = prices
            raw_obs[1::3] = quantities

        return raw_obs

    @property
    def handcrafted_market_features(self):
        """
        Returns all handcrafted features 1D np array.
        """
        crafted_obs = np.array([self.high_activity_flag,
                                self.relative_spread_obs(),
                                self.normed_midpoint_obs,
                                self.normed_lob_imbalance_obs,
                                self.normed_midpoint_moving_avg_obs,
                                self.normed_midpoint_moving_std_obs,
                                ])
        return crafted_obs

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
        # Note: time and inv are already normalized in agent_features.
        time = self.agent_features.elapsed_time
        inv = self.agent_features.remaining_inventory
        time_since_last_sub = \
            self.agent_features.time_since_last_submission_norm
        time_since_last_trade = \
            self.agent_features.time_since_last_trade_norm
        agent_obs = np.array([time, inv, time_since_last_sub,
                              time_since_last_trade])
        return agent_obs

    @property
    def high_activity_flag(self):
        """Return high activity flag."""
        return AgentContext.high_activity_flag

    def relative_spread_obs(self, scaling_factor=100):
        """Return relative spread observation"""
        rel_spread = self.market_features.rel_spread()
        return min(rel_spread*scaling_factor, 1)

    @property
    def normed_midpoint_obs(self):
        """Returns the normalized midpoint as price indicator."""
        midpoint = self.market_features.midpoint()
        normed_midpoint = (midpoint - MinMaxValues.min_price) / (
                MinMaxValues.max_price - MinMaxValues.min_price)
        return normed_midpoint

    @property
    def normed_midpoint_moving_avg_obs(self):
        """Returns the normalized midpoint moving avg as price indicator."""
        midpoint = self.market_features.midpoint_moving_avg()
        normed_midpoint = (midpoint - MinMaxValues.min_price) / (
                MinMaxValues.max_price - MinMaxValues.min_price)
        return normed_midpoint

    @property
    def normed_midpoint_moving_std_obs(self):
        """Moving standard deviation of midpoint"""
        mov_std = self.market_features.midpoint_moving_std()
        # I scale the STD by dividing by the midpoint and clip it to 1.
        return min(mov_std/self.market_features.midpoint(), 1)

    @property
    def normed_lob_imbalance_obs(self):
        """QB is naturally in the interval [-1, 1], I scale it to [0, 1]"""
        imb = self.market_features.lob_imbalance()
        imb_norm = (imb+1)/2
        return round(imb_norm, 3)

    def reset(self):
        """
        Note: the ObservationSpace must be composed freshly in agent every
        time the agent gets resetted, this resets the ObservationSpace
        and the compositions inside ObservationSpace automatically.
        """
        pass

