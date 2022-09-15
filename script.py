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
    #print("Episode Len: ", replay.episode.__len__())

    for i in range(replay.episode.__len__()):

        #print(i)
        replay.step()













