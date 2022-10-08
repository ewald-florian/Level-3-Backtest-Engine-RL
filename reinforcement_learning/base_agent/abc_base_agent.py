#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
""" Base RL Agent to be subclassed by RL agents for fast prototyping"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-09-23"
# version ='1.0'
# ----------------------------------------------------------------------------
#TODO: Is this abc class actually useful or should I just directly implement
#  the agent...

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

    # TODO: what is the best way to reset the agent when a new episode starts?
    #  probably, it is better to reset the subclass...
    #  the agent is instantiated inside replay, it will get reset inside the
    #  replay reset methods, the best approach would be to reset MarketInterface,
    #  Observation Space and Reward first and then instantiate a new agent instance
    #  with the resetted objects... (difficult if agent gets instantiated and then
    #  passed to replay...

    @abstractmethod
    def reset(self):
        """
        Reset Rl-Agent.
        """
        raise NotImplementedError("Impl. reset in special agent class")


