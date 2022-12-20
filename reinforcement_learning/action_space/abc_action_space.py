#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Abstract Action Space class for RL-Agent
"""
#----------------------------------------------------------------------------
__author__ = 'florian'
__date__ =  '12-20-2022'
__version__ = '0.1'
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

from market.market_interface import MarketInterface


class BaseActionSpace(ABC):
    """
    BaseActionSpace is an abstract class to be subclassed by a specific
    action space. The abstract method take_action must be implemented in the
    subclass. Standard action methods can be stored in the BaseActionSpace.
    """

    # start_date to compute latest min max prices
    def __init__(self):
        """
        Usually, BaseActionSpace is initialized via super in
        the respective ActionSpace. Initialization builds a composition
        of MarketInterface.
        """
        self.market_interface = MarketInterface()

    @abstractmethod
    def take_action(self, action):
        """
        Takes action as input and translates it into submissions or
        cancellations via market_interface.
        """
        raise NotImplementedError("Implement market_observation in subclass.")

    def reset(self):
        """
        Note: the ActionSpace must be composed freshly in agent every
        time the agent gets resets, this resets the ActionSpace
        and the compositions inside ActionSpace automatically.
        """
        pass

