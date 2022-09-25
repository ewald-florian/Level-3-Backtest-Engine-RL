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

from reinforcement_learning.environment import Environment
from reinforcement_learning.rl_agents.sample_agent import RlAgent
from replay.replay import Replay
from market.market import Market

if __name__ == '__main__':
    # instantiate agent
    agent = RlAgent()
    # instantiate replay and pass agent object as input argument
    replay = Replay(rl_agent=agent)

    # store replay instance in config dict
    config = {}
    config["env_config"] = {
        "config": {
            "replay": replay},
    }
    # instantiate and reset environment
    env = Environment(env_config=config)
    env.reset()

    # run loop
    number_of_steps = 20000
    training_list = []
    for step in range(10):
        # take a random action
        action = np.random.randint(3)
        # call env.step
        observation, reward, done, info = env.step(action) #calls replay.rl_step(action)
        # track activity
        store = [observation, reward, done, info]
        training_list.append(store)




