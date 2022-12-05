#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent Context to store important attributes"""

__author__ = "florian"
__date__ = "2022-11-13"
__version__ = "0.1"


class AgentContext:
    """Class to store relevant Agent-Parameters in class attributes.

    AgentContext can be used in Agent to update the class attributes and
    store the parameters. Afterwards, the parameters can be accessed by
    AgentFeatures to compute features based on them.

    Note: Since the update-methods are classmethods, it is possible to
    update the class attributes even without instantiating the class before.
    """

    # class attributes
    start_time = None
    end_time = None
    episode_length = None
    initial_inventory = None

    def __init__(self):
        """
        The class does not take any inputs.
        """
        pass

    @classmethod
    def update_start_time(cls, start_time):
        """Update class attribute start_time"""
        cls.start_time = start_time

    @classmethod
    def update_end_time(cls, end_time):
        """Update class attribute end_time"""
        cls.end_time = end_time

    @classmethod
    def update_episode_length(cls, episode_length):
        """Update class attribute episode_length"""
        cls.episode_length = episode_length

    @classmethod
    def update_initial_inventory(cls, initial_inventory):
        """Update class attribute initial_inventory"""
        cls.initial_inventory = initial_inventory

    @classmethod
    def reset(cls):
        """
        Reset class attributes.
        NOTE: AgentContext has to be reset before the Agent! However,
        since the values will be overwritten, reset is not necessary.
        """
        cls.start_time = None
        cls.end_time = None
        cls.episode_length = None
        cls.initial_inventory = None