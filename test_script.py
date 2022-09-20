#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# --------------------------------------------------------------------------
""" Test and debug Level-3 backtest engine"""
# --------------------------------------------------------------------------
from replay.replay import Replay
from market.market import Market
from market.context import Context
from feature_engineering.market_features import MarketFeatures
from reinforcement_learning.observation_space import ObservationSpace
from reinforcement_learning.reward import Reward
from market.market_trade import MarketTrade
from market.market_interface import MarketInterface
from agent.agent_metrics import AgentMetrics

# Dev MarketFeatures
if __name__ == '__main__':
    replay = Replay()
    replay.reset()

    mf = MarketFeatures()
    obs_space = ObservationSpace()
    reward = Reward()
    am = AgentMetrics()
    mi = MarketInterface()

    for i in range(replay.episode.__len__()):

        replay.step()
        # collect context
        Context(Market.instances['ID'].state_l3)
        # generate orders
        if i % 1000 == 0:
            limit = mf.best_bid()
            mi.submit_order(side=2, quantity=2220000, limit=limit)

        if i % 1000 == 500:
            limit = mf.best_ask()
            mi.submit_order(side=1, quantity=2220000, limit=limit)

        # AgentMetrics FEBUGGING
        print("filtered messages", am.get_filtered_messages(side=2))
        print("fm static", am.get_filtered_messages_static(side=2))
        print("gft", am.get_filtered_trades(side=2))
        print("get realized trades:", am.get_realized_trades)
        print("uq", am.unrealized_quantity)
        print("rq", am.realized_quantity)
        print("pv", am.position_value)
        print("exp", am.exposure)
        print("cp", am.cash_position)
        print("gzt", am.get_unrealized_trades)
        print("VWAP buy", am.vwap_buy)
        print("VWAP sell", am.vwap_sell)
        print("vscore", am.vwap_score)
        print("pnl_real", am.pnl_realized)
        print("pnl_unreal", am.pnl_unrealized)
        print("exp_left", am.exposure_budget_left)
        print("tc", am.transaction_costs)
        print(am) # string repesentation
        print()
        print()























