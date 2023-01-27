#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Context Module for the Level-3 backtest engine:
Stores Market States which can then be used to
generate features.
"""
# ---------------------------------------------------------------------------

#TODO: maybe it would be an interesting option to no save every single
# state but instead catch a longer time window in a shorter list, i.e.
# leave out some states in between...


class Context:

    # class attribute
    context_list = list()
    midpoints = list()

    def __init__(self,
                 market_state: dict,
                 context_length: int = 100):
        """
        Context class maintains the context list as class attribute
        which can be used to store market states. The context_list can be
        used to generate features which require a timeseries of market states.
        Context can potentially store various representations in context_list,
        e.g. level1, level2, level3 states or all of them combined. Context
        should be reset before each new episode.

        The designated was to use Context is:
          current_state = Market.state_l3
          Context(current_state)

        Several state representations can be stored together, e.g. in a dict:
          state_l1 = Market.state_l1
          state_l2 = Market.state_l2
          state_l3 = Market.state_l3

          current_state_dict = {L1:state_l1, L2:state_l2, L3:state_l3}

        :param market_state,
            market_state which should be stored, can take different dtypes
        :param context_length,
            int, length of context_list
        """

        # attributes from arguments
        self.market_state = market_state
        self.context_length = context_length

        # global attributes update
        self.__class__.context_list.append(self.market_state)
        # remove older market states depending on context length
        self.__class__.context_list = self.__class__.context_list[
                                      -self.context_length:]

        # Collect midpoint
        self.__class__.midpoints.append(int(self.compute_midpoint))
        self.__class__.midpoints = self.__class__.midpoints[
                                      -self.context_length:]

    def __str__(self):
        """
        String representation.
        """
        pass

    @property
    def compute_midpoint(self):
        best_bid = max(self.market_state[1].keys())
        best_ask = min(self.market_state[2].keys())
        midpoint = (best_bid + best_ask) / 2
        return midpoint

    @classmethod
    def reset_context_list(cls):
        """
        Reset context list.
        """
        del cls.context_list[:]


