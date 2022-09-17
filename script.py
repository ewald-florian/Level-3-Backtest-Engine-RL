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
from agent.market_interface import MarketInterface
import time

if __name__ == '__main__':
    #start = time.time()

    replay = Replay()
    replay.reset()
    print("Episode Len: ", replay.episode.__len__())
#
    for i in range(replay.episode.__len__()):
        replay.step()

    #end = time.time()
    #print("TIME: ", end-start)














