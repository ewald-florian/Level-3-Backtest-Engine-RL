#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
""" Base RL Agent to be subclassed by RL agents for fast prototyping"""

from abc import ABC, abstractmethod


class RlBaseAgent(ABC):
    """
    Base class for RL agents to be subclassed by RL agents. This abstract
    class just dictates a structure for the special agent class and does not
    contain any real functionality. The abstractmethods to be implemented in
    the subclass are:

    - step()
    - take_action()
    - reset()
    """
    def __init__(self):
        """
        Class has no attributes.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Implement the agent step. This includes to execute an action,
        make an observation, receive a reward and return the observation
        and the reward. Take action is called inside step() such that the
        step method can be used as entry point to the RL agent.
        """
        raise NotImplementedError("Implement step in special agent class")

    @abstractmethod
    def _take_action(self, action):
        """
        Takes action as input and executes it. This typically include specifying
        order details and making a submission or cancellation via the market
        interface.
        """
        raise NotImplementedError("Impl. take_action in special agent class")

    @abstractmethod
    def reset(self):
        """
        Reset Rl-Agent.
        """
        raise NotImplementedError("Impl. reset in special agent class")


