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
from market.market_state_v1 import Market
from agent.market_interface import MarketInterface


if __name__ == '__main__':

    replay = Replay()
    replay.reset()
    print("Episode Len: ", replay.episode.__len__())
#
    for i in range(1000):#replay.episode.__len__()):
#
        print(i)
        replay.step()














