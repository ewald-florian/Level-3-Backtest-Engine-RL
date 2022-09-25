#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 21/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Step the RL environment with random action for developing, testing and
debugging.
"""

import numpy as np


from reinforcement_learning.environment import Environment
from reinforcement_learning.rl_agents.sample_agent import RlAgent
from replay.replay import Replay

from market.market import Market


# CHANGE AGENT IM REPLAY

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


    env = Environment(env_config=config)
    env.reset()

    number_of_steps = 20000
    training_list = []

    # loop
    for step in range(10):


        # take a random action
        action = np.random.randint(3)

        observation, reward, done, info = env.step(action) #calls replay.rl_step(action)

        #store = [observation, reward, done, info]
        #training_list.append(store)

        # check whats going on inside the environment
        #print(Market.instances['ID'].state_l1)



