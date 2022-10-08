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
import numpy as np
import gym
from gym import spaces

from reinforcement_learning.action_space.action_storage import ActionStorage
from reinforcement_learning.transition.agent_transition import AgentTransition
from reinforcement_learning.transition.env_transition \
    import EnvironmentTransition


class Environment(gym.Env):
    """
    Environment class for Reinforcement Learning with the Level-3 Backtest
    Engine. Follows OpenAI gym convention.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 env_config: dict = None):
        """
        Set observation-space and action-space. Instantiate Replay.
        """

        # TODO: obtain spaces from env_config
        self.action_space_arg = 3
        # self.observation_space_min = np.zeros(30) #minmax
        self.observation_space_min = np.array([-1000_000] * 30)
        self.observation_space_max = np.array([1000_000] * 30)

        self.action_space = spaces.Discrete(self.action_space_arg)
        self.observation_space = spaces.Box(self.observation_space_min,
                                            self.observation_space_max)

        # Entry Points to Level-3 Backtest Engine:

        # -- instantiate replay_episode to step the environment

        # 1. For Rllib: rllib unpacks the config dict automatically...
        try:
            self.replay = env_config.get("config").get("replay_episode")
        # 2. For other loops (e.g. toy_training_loop)
        except:
            self.replay = env_config['env_config']['config']['replay_episode']

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

        # pass action to agent via ActionStorage class attribute
        ActionStorage(action)
        #print("(ENV) action storage: ", ActionStorage.action)
        # replay step (now, without action)
        self.replay.rl_step()

        # get AgentTransition
        observation, reward = AgentTransition.transition
        #print("(ENV) AgentTransition.transition: ", AgentTransition.transition)

        # get EnvironmentTransition
        done, info = EnvironmentTransition.transition
        #print("(ENV) EnvironmentTransition.transition: ", EnvironmentTransition.transition)
        """ # old version
        # -- Take step and receive observation, reward, info
        observation, reward, done, info = self.replay.rl_step(action)
        # -- Return
        """
        # return
        return observation, reward, done, info

    def reset(self):
        """
        Reset Environment to initial state. Returns the first observation of
        the environment. Reset has to be called at the beginning of each
        episode.
        """
        # -- reset replay_episode
        first_obs = self.replay.rl_reset()

        print('(ENV)  episode len: ', self.replay.episode.__len__())

        return first_obs

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
