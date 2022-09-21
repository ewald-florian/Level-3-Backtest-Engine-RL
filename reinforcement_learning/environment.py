#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Implementation of the reinforcement environment.
"""
# ---------------------------------------------------------------------------

import gym
from gym import spaces

from replay.replay import Replay
from market.market_interface import MarketInterface


class Environment(gym.Env):
    """
    Environment class for Reinforcement Learning with the Level-3 Backtest
    Engine. Follows OpenAI gym convention.
    """

    def __init__(self,
                 env_config: dict):
        """
        Set observation-space and action-space. Instantiate Replay.
        """

        # TODO: obtain spaces from env_config
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.zeros(52),
                                            np.array([100_000_000] * 52))

        # Entry Points to Level-3 Backtest Engine
        # -- replay to step the environment
        self.replay = Replay()
        # -- market_interface to execute actions
        # TODO: build ActionSpace as intermediary to translate policy actions
        #  into submissions/cancellations
        self.market_interface = MarketInterface()

    def step(self, action):
        """
        Executes a step in the environment by applying the action.
        Transitions the environment to the next state. Returns the new
        observation, reward, completion status (done), and additional info.

        :param action:
            ...
        :return: observation
            np.array, ...
        :return: reward
            float, ...
        :return: done
            bool, True if episode is complete, False otherwise
        :return: info
            dict, additional info, can be empty
        """

        # assert if action is valid
        assert self.action_space.contains(action), "Invalid Action"

        # -- 1.) Take action

        self.market_interface ...

        # -- 2.) Take step and receive observation, reward, info

        observation, reward, done, info = self.replay ...

        # -- 3.) Return

        return observation, reward, done, info



    def reset(self):
        """
        Reset Environment to initial state. Returns the first observation of
        the environment. Reset has to be called at the beginning of each
        episode.
        """
        # -- reset replay
        self.replay.reset()

        # -- reset market interface
        self.market_interface.reset()

    def render(self):
        """
        Render the environment.
        """
        assert mode in ["human"], "Invalid mode"
        # TODO
        if mode == "human":
            pass


    def seed(self):
        """
        Set random seed for the environment.
        """
        pass

