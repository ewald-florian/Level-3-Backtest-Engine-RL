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

from agent.agent_order import OrderManagementSystem as OMS
from reinforcement_learning.environment import Environment
from market.market import Market

if __name__ == '__main__':

    env = Environment()
    env.reset()

    number_of_steps = 20000
    training_list = []

    # loop
    for step in range(1000):


        # take a random action
        action = np.random.randint(3)
        print(action)

        observation, reward, done, info = env.step(action)

        store = [observation, reward, done, info]
        training_list.append(store)

        print(OMS.order_list)

