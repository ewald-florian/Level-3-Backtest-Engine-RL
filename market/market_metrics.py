#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market metrics computed based on execution summaries during the
backtest.
"""

from market.market_trade import MarketTrade


class MarketMetrics:
    """
    Class to compute relevant market metrics based on MarketTrade.history
    which contains a list of all historical executions during the episode.
    """

    def __init__(self):
        pass

    @property
    def market_vwap_sell(self):
        """Compute the market vwap since episode start."""
        market_sell_vwap = 0
        # Check if market sell orders exist.
        if len([t for t in MarketTrade.history if t['side'] == 2]) > 0:
            sell_volume = sum([t['quantity'] for t in MarketTrade.history if
                               t['side'] == 2])
            price_times_volume = sum([t['quantity']*t['price'] for t in
                                MarketTrade.history if t['side'] == 2])
            market_sell_vwap = price_times_volume / sell_volume

        return market_sell_vwap

    @property
    def market_vwap_buy(self):
        """Compute the market vwap since episode start."""
        market_buy_vwap = 0
        # Check if market sell orders exist.
        if len([t for t in MarketTrade.history if t['side'] == 1]) > 0:
            sell_volume = sum([t['quantity'] for t in MarketTrade.history if
                               t['side'] == 2])
            price_times_volume = sum([t['quantity'] * t['price'] for t in
                                      MarketTrade.history if t['side'] == 2])
            market_buy_vwap = price_times_volume / sell_volume

        return market_buy_vwap

    def trading_volume(self):
        """Compute the historical trading volume since episode start."""
        pass

    def net_traded_flow(self):
        """
        Flow is the net traded volume, i.e. buy volume minus sell volume.
        To be used as momentum and market sentiment factor.
        """
        pass

    def reset(self):
        """
        Reset MarketMetrics
        """
        self.__init__()