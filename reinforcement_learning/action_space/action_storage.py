#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ---------------------------------------------------------------------------
""" Storage Class to deliver action from Env to Agent"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-09-23"
__status__ = "production"
__version__ = "0.1"
# ---------------------------------------------------------------------------


class ActionStorage:
    """
    Storage to temporarily store an action in order
    to deliver it from the policy to an rl agent.
    """
    action = None

    @classmethod
    def __init__(cls, action=None):
        """
        Stores action to class attribute 'action'
        :param action
            any, current action to be executed by the agent
        """
        cls.action = action

    @classmethod
    def reset(cls):
        """
        Reset trade history.
        """
        # delete all elements in Trade.history (list)
        cls.action = None
