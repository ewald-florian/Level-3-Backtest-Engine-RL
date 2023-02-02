#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Implementation of the reinforcement learning environment.
"""
# ---------------------------------------------------------------------------

import copy
import json

import numpy as np
import gym
from gym import spaces

from reinforcement_learning.action_space.action_storage import ActionStorage
from reinforcement_learning.transition.agent_transition import AgentTransition
from reinforcement_learning.transition.env_transition \
    import EnvironmentTransition

# For Episode Stats.
from agent.agent_order import OrderManagementSystem as OMS
from agent.agent_trade import AgentTrade
from reinforcement_learning.environment.episode_stats import EpisodeStats
from reinforcement_learning.action_space.action_storage import ActionStorage


class TradingEnvironment(gym.Env):
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
        # Get obs and action sizes from config dict.
        observation_size = env_config['observation_size']
        action_size = env_config['action_size']

        # Define observation space.
        self.observation_space_min = np.array([-10_000_000] * observation_size)
        self.observation_space_max = np.array([10_000_000] * observation_size)
        self.observation_space = spaces.Box(self.observation_space_min,
                                            self.observation_space_max)

        # Define action space
        self.action_space = spaces.Discrete(action_size)

        # Entry Points to Level-3 Backtest Engine:

        # -- instantiate replay_episode to step the environment

        # 1. For Rllib: rllib unpacks the config dict automatically...
        try:
            self.replay = env_config.get("config").get("replay_episode")
        # 2. For other loops (e.g. toy_training_loop)
        except:
            self.replay = env_config['env_config']['config']['replay_episode']

        # DEBUGGING
        self.step_counter = 0

    def step(self, action):
        """
        Executes a step in the environment by applying the action.
        Transitions the environment to the next state. Returns the new
        observation, reward, completion status (done), and additional info.

        :param action:
            int, action obtained from Model.
        :return: observation
            np.array, observation
        :return: reward
            float, reward
        :return: done
            bool, True if episode is complete, False otherwise
        :return: info
            dict, additional info, can be empty
        """
        # assert if action is valid
        assert self.action_space.contains(action), "Invalid Action"

        # pass action to agent via ActionStorage class attribute
        ActionStorage(action)

        # replay step (now, without action)
        self.replay.rl_step()

        # get AgentTransition
        observation, reward = AgentTransition.transition

        # get EnvironmentTransition
        done, info = EnvironmentTransition.transition

        # If Episode is done, store epsiode stats.
        if done:
            oms = copy.deepcopy(OMS.order_list)
            agent_trade = copy.deepcopy(AgentTrade.history)
            action_list = ActionStorage.action_history
            #EpisodeStats.store_episode_results(oms, agent_trade, action_list)

        # return
        return observation, reward, done, info

    def reset(self):
        """
        Reset Environment to initial state. Returns the first observation of
        the environment. Reset has to be called at the beginning of each
        episode.
        """
        # Reset replay_episode.
        first_obs = self.replay.rl_reset()

        return first_obs

    def render(self):
        """
        Render the environment.
        """
        pass

    def seed(self):
        """
        Set random seed for the environment.
        """
        pass
