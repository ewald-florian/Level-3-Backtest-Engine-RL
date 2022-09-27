#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ---------------------------------------------------------------------------
"""
Step the RL environment with random action for developing, testing and
debugging. Does not include Reinforcement Learning or any form of ML.
"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-09-21"
__version__ = "0.1"
# ----------------------------------------------------------------------------

import numpy as np

from reinforcement_learning.environment.environment import Environment
from reinforcement_learning.rl_agents.sample_agent import RlAgent
from replay_episode.replay import Replay
from market.market import Market

if __name__ == '__main__':
    # instantiate agent
    agent = RlAgent(verbose=False)
    # instantiate replay_episode and pass agent object as input argument
    replay = Replay(rl_agent=agent)

    # store replay_episode instance in config dict
    config = {}
    config["env_config"] = {
        "config": {
            "replay_episode": replay},
    }
    # instantiate and reset environment
    env = Environment(env_config=config)
    env.reset()

    # run loop
    number_of_steps = 20000
    training_list = []

    print("LEN EPISODE: ", replay.episode.__len__())

    # run a single episode
    for step in range(replay.episode.__len__()):
        print(step)
        # take a random action
        action = np.random.randint(3)
        # call env.step
        observation, reward, done, info = env.step(action) #calls replay_episode.rl_step(action)
        # track activity
        #store = [observation, reward, done, info]
        #training_list.append(store)

    print('...loop finished')




