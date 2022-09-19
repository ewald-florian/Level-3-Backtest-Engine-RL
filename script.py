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

# Dev MarketFeatures
if __name__ == '__main__':
    replay = Replay()
    replay.reset()

    mf = MarketFeatures()
    obs_space = ObservationSpace()
    reward = Reward()

    for i in range(replay.episode.__len__()):
        replay.step()

        state_l3 = Market.instances['ID'].state_l3

        Context(state_l3)

        array = mf.level_2_plus(data_structure='array', store_hhi=False, store_timestamp=False)
        print(array)

        #print(obs_space.market_observation())
        if MarketTrade.history:
            print(MarketTrade.history)

        MarketInterface.submit_order(side=1, limit=30000000000, quantity=500000)





"""
if __name__ == '__main__':
    #start = time.time()

    replay = Replay()
    replay.reset()
    print("Episode Len: ", replay.episode.__len__())
#
    for i in range(replay.episode.__len__()):
        replay.step()

"""














