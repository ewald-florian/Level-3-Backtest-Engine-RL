#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MinMaxPriceStorage to store min and max for episode normalization."""

__author__ = "florian"
__date__ = "2022-11-29"
__version__ = "0.1"


class MinMaxPriceStorage:
    """Class to store minimum and maximum prices for normalization.

    This is a storage class. The min and max prices at the beginning
    of each episode are stored in this class as class attributes. The initial
    min and max prices can for example be the level-20 bid and ask price which
    can be retrieved during the episode build in the episode class.
    Moreover, these prices are decreased/increased by a deviation factor,
    for example of 20% to ensure that the price range is not crossed during
    the episode. The deviation factor has to be adjusted manually in the
    classmethods since the class does not need to be instantiated to be used
    as storage.

    Note: Since the update-methods are classmethods, it is possible to
    update the class attributes even without instantiating the class before.
    """

    # class attributes
    max_price = None
    min_price = None

    def __init__(self):
        """
        The class does not take any inputs.
        """
        pass

    @classmethod
    def update_min_price(cls, initial_min_price):
        """The initial minprice is adjusted with an additional deviation
        factor, e.g. 20% to make sure that the price band will not be
        crossed during the episode
        :param initial_min_price
            int, initial min price
        """
        cls.min_price = initial_min_price * 0.8

    @classmethod
    def update_max_price(cls, initial_max_price):
        """The initial maxprice is adjusted with an additional deviation
        factor, e.g. 20% to make sure that the price band will not be
        crossed during the episode
        :param initial_max_price
            int, initial max price of episode
        """
        cls.max_price = initial_max_price * 1.2

    @classmethod
    def reset(cls):
        """
        Reset class attributes.
        """
        cls.max_price = None
        cls.min_price = None