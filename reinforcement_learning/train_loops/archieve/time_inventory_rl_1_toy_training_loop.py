#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
AGENTID = 161122

Step the RL environment with random action for developing, testing and
debugging. Does not include Reinforcement Learning or any form of ML.
"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-11-13"
__version__ = "0.1"
# ----------------------------------------------------------------------------

import numpy as np

from reinforcement_learning.environment.tradingenvironment import TradingEnvironment
from replay_episode.replay import Replay
from market.market import Market

from reinforcement_learning.agent_prototypes.archieve.time_inventory_rl_agent import \
    TimeInventoryAgent1

if __name__ == '__main__':
    # instantiate agent
    #agent = RlAgent(verbose=True)
    agent = TimeInventoryAgent1()
    # instantiate replay_episode and pass agent object as input argument
    replay = Replay(rl_agent=agent,
                    episode_length="1m")

    # store replay_episode instance in config dict
    config = {}
    config["env_config"] = {
        "config": {
            "replay_episode": replay},
    }
    # instantiate and reset environment
    env = TradingEnvironment(env_config=config)
    env.reset()

    # run loop
    number_of_steps = 5
    training_list = []

    print("LEN EPISODE: ", replay.episode.__len__())

    # run a single episode
    for step in range(replay.episode.__len__()):
        print(step)
        # take a random action
        action = np.random.randint(2)
        # call env.step
        observation, reward, done, info = env.step(action) #calls replay_episode.rl_step(action)
        print('(LOOP) DONE: ', done)
        # track activity
        #store = [observation, reward, done, info]
        #training_list.append(store)
        print("Market: ", Market.instances['ID'].state_l1)

    print('...loop finished')




