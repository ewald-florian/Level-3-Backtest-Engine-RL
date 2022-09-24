#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
""" Base RL Agent to be subclassed by RL agents for fast prototyping"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-09-23"
# version ='1.0'
# ----------------------------------------------------------------------------

# TODO: Idee weiter verfolgen, debuggen, testen
#  überlegen welche Struktur am meisten Sinn ergibt...(z.B. auch
#  ObservationSpace und reward abstrahieren für maximal schnelles prototyping
import abc

from market.market_interface import MarketInterface
from reinforcement_learning.observation_space import ObservationSpace
from reinforcement_learning.reward import Reward


class RlBaseAgent(abc.ABC):
    """
    Base class for RL agents to be subclassed by RL agents. Responsible for
    the steady tasks such as initialization and reset which are equal for every
    kind of RL agent. The specific methods

    - step()
    - take_action()

    have to be implemented in the respective specialized agent
    which subclasses RlBaseAgent.
    """

    def __init__(self):
        """
        Initialize base agent.
        """

        # create class instances
        self.market_interface = MarketInterface()
        self.reward = Reward()
        self.observation_space = ObservationSpace()

    @abc.abstractmethod
    def step(self, action):
        """
        Implement the agent step. This includes to exeecute an action,
        make an observation, receive a reward and return the observation
        and the reward. Take action is called inside step() such that the
        step method can be used as entry point to the RL agent.
        """
        raise NotImplementedError("Implement step in specialized agent class")

    @abc.abstractmethod
    def _take_action(self, action):
        """
        Takes action as input and executes it. This typically include specifying
        order details and making a submission or cancellation via the market
        interface.
        """
        raise NotImplementedError("Implement take action in specialized "
                                  "agent class")

    def reset(self):
        """
        Reset RlBaseAgent.
        """
        self.__init__()


# The Agent inherits the Base Agent
from reinforcement_learning.rl_agents.rl_base_agent import RlBaseAgent


class RlAgentTest(RlBaseAgent):

    def __init__(self,
                 quantity: int = 10_0000):

        # static
        super().__init__()
        self.quantity = quantity

    def step(self, action):

        # take action
        self._take_action(action)

        # make observation
        observation = self.observation_space

        # receive reward
        reward = self.reward.pnl_marginal

        # return

        return observation, reward

    # ActionSpace
    def _take_action(self, action):

        best_ask = self.market_features.best_ask()
        best_bid = self.market_features.best_bid()

        # buy
        if action == 1 and best_ask:
            self.market_interface.submit_order(side=1,
                                               limit=best_ask,
                                               quantity=self.quantity)
            print('buy submission')
        # sell
        elif action == 2 and best_bid:
            self.market_interface.submit_order(side=2,
                                               limit=best_bid,
                                               quantity=self.quantity)
            print('sell submission')
        # wait
        else:
            pass

