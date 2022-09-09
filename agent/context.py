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
# TODO: Checken ob ich deepcopy für den market state brauche...
# TODO: Evtl. die trade execution (13202) history in context Speichern? Um Market VWAP/TWAP etc zu berechnen...
class Context:

    # class attribute
    context_list = list()

    def __init__(self,
                 market_state,
                 context_length:int=55):
        """
        Context class maintains the context list as class attribute
        which can be used to store market states by calling
        Context(market_state). The context_list can be used to generate
        features wich require a timeseries of market states. Context can
        potentially store various representations in context_list, e.g. level1,
        level2, level3 states or all of them combined. Context should be resetted
        before each new episode.

        :param market_state,
            market_state which should be stored, can take different dtypes
        :param context_length,
            int, defines number of market_states which are stored in context_list
        """

        # attributes from arguments
        self.market_state = market_state
        self.context_length = context_length

        # global attributes update
        self.__class__.context_list.append(self.market_state)
        # remove older market states depending on context length
        self.__class__.context_list = self.__class__.context_list[-self.context_length:]

    def __str__(self):
        """
        String representation.
        """
        pass

    @classmethod
    def reset_context_list(class_reference):
        """
        Reset context list.
        """
        del class_reference.context_list[:]


