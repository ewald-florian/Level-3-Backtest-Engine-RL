#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 17/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Class to compute MarketMetrics
"""
# ---------------------------------------------------------------------------

# TODO: It might be interesting to compute these metrics for the entire trading
#  day and not just for the episode... (for general purposes, not necessarily
#  for rl)
# AgentTrade ist nur eine Liste, hier werden die metrics berechnet


class MarketMetrics:

    def __init__(self):
        pass

    # TODO: most important, relevant for other features
    def market_vwap(self):
        pass

    def trading_volume(self):
        pass

    def net_traded_flow(self):
        """
        Flow is the net traded volume, i.e. buy volume minus sell volume.
        To be used as momentum and market sentiment factor.
        """
    def reset(self):
        pass