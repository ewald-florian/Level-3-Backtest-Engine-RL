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

if __name__ == '__main__':

    replay = Replay()
    replay.reset()
    print("Episode Len: ", replay.episode.__len__())

    for i in range(replay.episode.__len__()):
        #print(i)
        replay.step()

























# -- Experiment to build MarketState as independent instance

#from environment.market_state_new import MarketStateAttribute
#import json

#if __name__ == '__main__':
#   #replay = Replay()
#    # reset builds a new episode
#    # replay.reset()
#    _ = MarketStateAttribute(market_id="ID_0")
#    PATH = "/Users/florianewald/PycharmProjects/Level3-Data-Analysis/sample_msg_data/DE0005190003.XETR_20220201T120211_20220201T163000"
#    snapshot_start_file = open(f"{PATH}/snapshot_start.json")
#    snapshot_start = json.load(snapshot_start_file)[0]
#    #print(snapshot_start)
#    # call the specific instance of MarketState...
#    #_.initialize_state(snapshot=snapshot_start)
#    MarketStateAttribute.instance.initialize_state(snapshot=snapshot_start)
#    print('###STATE###')
#    print(MarketStateAttribute.instance.state_l1)
