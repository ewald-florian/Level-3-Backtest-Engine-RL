#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
"""Storage class to deliver reward and observation from Agent to Env"""


class EnvironmentTransition:
    """
    Storage class to deliver done and info from Replay to Env.
    """
    # class attribute tuple
    transition = ()

    @classmethod
    def __init__(cls,
                 done,
                 info):
        """
        Stores observation and reward to the agent_transition class attribute
        tuple.
        :param done
            bool, True in env is done
        :param info
            dict, additional env information, can be left empty
        """
        cls.transition = (done, info)

    @classmethod
    def reset(cls):
        """
        Reset EnvironmentTransition by setting the class attribute to an
        empty tuple.
        """
        cls.transition = ()