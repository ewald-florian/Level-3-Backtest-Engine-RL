#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
TEMPLATE for RL-Agent Prototypes.
---------------------------------
A Prototype of a 'Special Agent' has to be build using a construct of
abstract- and subclasses. The SpecialAgent is then controlled from replay.

- ObservationSpace subclasses BaseObservationSpace, the abstract methods which
    need to be implemented are agent_observation and market_observation.
- Reward subclasses BaseReward, the abstract method which needs to be
    implemented is receive_reward.
- SpecialAgent subclasses RLBaseAgent, the abstract methods which need to
    be implemented are step, _take_action and reset.
- SpecialAgent builds compositions of ObservationSpace and Reward for RL and
    MarketInterface to execute actions.

The first goal of this architecture is to allow fast efficient prototyping. The
sole requirement for a new prototype is the respective implementation of the
abstract methods. The second goal is sustainable archiving of prototypes. Each
prototype can be stored in one piece as a single py file containing the
implemented subclasses ObservationSpace, Reward and >SpecialAgent<.

This template can be copied to implement a new prototype. Prototypes should
have a unique and meaningful name and be located in the agent_prototypes
package.
"""

from copy import copy

import numpy as np
import pandas as pd

from market.market_interface import MarketInterface
from feature_engineering.market_features import MarketFeatures
from reinforcement_learning.reward.abc_reward import BaseReward
from reinforcement_learning.observation_space.abc_observation_space \
    import BaseObservationSpace
from reinforcement_learning.base_agent.abc_base_agent import RlBaseAgent
from reinforcement_learning.transition.agent_transition import AgentTransition
from reinforcement_learning.action_space.action_storage import ActionStorage


class ObservationSpace(BaseObservationSpace):
    """
    Subclass of BaseObservationSpace to implement the observation for a
    specific agent. The abstract methods market_observation and
    agent_observation need to be implemented.

    Note:
    ----
    The returns of market_obs and agent_obs are concatenated in the
    method holistic_observation inherited from the parent class which can
    be directly called by the rl-agent to receive the holistic observation.

    Normalization has to be be defined inside the abstract methods. The parent
    class provides some helpful normalization methods which can be used for
    that step.
    """

    def __init__(self):
        """
        Initiate parent class with super function. This may be relevant for
        normalization parameters such as min-max prices and quantities.
        """
        super().__init__()

    def market_observation(self) -> np.array:
        """
        Implement the market observation, the public part of the agents
        observation space.
        """
        # -- market features
        market_obs = ...
        return market_obs

    def agent_observation(self) -> np.array:
        """
        Implement the agent observation. The private part of the agents
        observation space.
        """
        agent_obs = ...
        return agent_obs


class Reward(BaseReward):
    """
    Subclass of base reward to implement the reward for specific agent.
    The abc method receive_reward needs to be implemented.
    """
    def __init__(self):
        """
        Instantiate parent class.
        """
        super().__init__()

    def receive_reward(self):
        """
        Returns reward. This abstract method should be implemented to
        specify the reward function for the specific agent. The reward
        function can either be new custom reward function or be based on the
        collection of standard reward functions in the parent class.
        :return reward
            ..., current reward for the rl-agent
        """
        reward = ...
        return reward


class SpecialAgent(RlBaseAgent):
    """
    Template for SpecialAgents which are based on specific reward and
    observation space which together with the specific take_action method
    constitute the characteristic of the special agent.

    Note:
    -----
    - SpecialAgent is a placeholder, change to meaningful name.
    - Each SpecialAgent can be stored in a single py file together with its
        corresponding Reward and ObservationSpace classes.
    - Write complementary documentation for the implemented methods of the
        SpecialAgent
    """
    def __init__(self,
                 verbose=True):
        """
        When initialized, SpecialAgent builds compositions of MarketInterface,
        Reward and ObservationSpace. Note that Reward and ObservationSpace
        are subclasses which should be implemented to meet the specific
        requirements of this special agent, a specific observation and a
        specific reward function.
        """
        # static
        super().__init__()
        self.verbose = verbose

        # compositions
        self.market_interface = MarketInterface()
        self.reward = Reward()
        self.observation_space = ObservationSpace()

        self.market_features = MarketFeatures()

    # TODO: this method does not really need to be adjusted I could almost
    #  add it to the abstract parent class.
    def step(self):
        """
        Step executes the action, gets a new observation, receives the reward
        and returns reward and observation.
        """
        # -- take action
        action = ActionStorage.action
        self._take_action(action)
        # -- make observation
        observation = copy(self.observation_space.holistic_observation())
        # -- receive reward
        reward = copy(self.reward.receive_reward())
        # -- pass agent transition
        AgentTransition(observation, reward)

    def _take_action(self, action):
        """
        Take action as input and translate into trading decision.
        :param action,
            ..., next action
        """
        pass

    def reset(self):
        """
        Reset SpecialAgent before each new episode. Basically, copy the
        __init__ method. Make sure to add all additional static and
        dynamic attributes.
        """
        # TODO: add all attributes of the special agent
        super().__init__()
        self.verbose = self.verbose

        # compositions
        self.market_interface.reset()
        self.reward.reset()
        self.observation_space.reset()
        self.market_features.reset()
