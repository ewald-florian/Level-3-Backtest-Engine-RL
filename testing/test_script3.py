#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------------
# Created By  : florian
# Created Date: 23/Sept/2022
# version ='1.0'
# --------------------------------------------------------------------------
""" Test and debug observation_space"""
# --------------------------------------------------------------------------
from replay_episode.replay import Replay
from market.market import Market
from context.context import Context
from feature_engineering.market_features import MarketFeatures
from reinforcement_learning.observation_space import ObservationSpace
from reinforcement_learning.reward import Reward
from market.market_interface import MarketInterface
from agent.agent_metrics import AgentMetrics

if __name__ == '__main__':
    replay = Replay()
    replay.rl_reset()

    mf = MarketFeatures()
    obs_space = ObservationSpace()
    reward = Reward()
    am = AgentMetrics()
    mi = MarketInterface()

    for i in range(10):#replay_episode.episode.__len__()):
        replay.rl_step_old()
        Context(Market.instances['ID'].state_l3)

        print(mf.best_ask())
        print(mf.best_bid())

    # DEBUGGING
    obs = obs_space.market_observation()