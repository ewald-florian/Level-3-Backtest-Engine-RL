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

from reinforcement_learning.market_features import MarketFeatures

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
        # -- update class attribute
        self.__class__.instance = self.holistic_observation()

    def market_observation(self):
        # -- market features
        market_obs = self.market_features.level_2_plus()

        # -- normalize


        return market_obs

    def _min_max_normalization(self, prices, quantities):
        """
        Scales values between 0 and 1.
        """
        #TODO: historische min-max values automatisch berechnen oder
        # vorher in csv oÄ Speichern (für Level 1/5/10/20)

        #TODO: Clip outliers, muss aber sinnvoll sein! am besten nochmal in
        # den ML/HFT Büchern nachlesen...

        scaled_prices = (prices - self.min_price)/(
                    self.max_price - self.min_price)
        # scale quantities
        scaled_quantities = (quantities - self.min_size)/(
                    self.max_size - self.min_size)

        return scaled_prices, scaled_quantities

    def _arbitrary_min_max_nomalization(self, a, b, prices, quantities):
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
