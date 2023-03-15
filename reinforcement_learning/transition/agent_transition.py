#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
"""Storage class to deliver reward and observation from Agent to Env"""


class AgentTransition:
    """
    Storage class to deliver reward and observation from Agent to Env.
    """
    # class attribute tuple
    transition = ()

    @classmethod
    def __init__(cls,
                 observation,
                 reward):
        """
        Stores observation and reward to the agent_transition class attribute
        tuple.
        :param observation
            np.array, latest observed observation
        :param reward
            float, latest received reward
        """
        cls.transition = (observation, reward)

    @classmethod
    def reset(cls):
        """
        Reset AgentTransition by setting the class attribute to an
        empty tuple.
        """
        cls.transition = ()