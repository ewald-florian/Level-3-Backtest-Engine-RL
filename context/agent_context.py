#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent Context to store important attributes"""

__author__ = "florian"
__date__ = "2022-11-13"
__version__ = "0.1"

import pandas as pd


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
    high_activity_flag = None

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
    def update_episode_length_ns(cls, episode_length_str):
        # Convert to pandas timestamp.
        episode_delta = pd.Timedelta(episode_length_str)
        # Convert to nanoseconds int.
        cls.episode_length = episode_delta.delta

    @classmethod
    def update_initial_inventory(cls, initial_inventory):
        """Update class attribute initial_inventory"""
        cls.initial_inventory = initial_inventory

    @classmethod
    def update_high_activity_flag(cls, start_time):
        """
        Update high activity flag which can be used as observation
        feature.
        """
        cls.high_activity_flag = 1

        start_time = pd.to_datetime(
            int(AgentContext.start_time), unit='ns')
        date = str(start_time.date()).replace("-", "")
        time = str(start_time.time()).replace("-", "").replace(":", "")
        # Check if it is summer or winter time.
        # Check if it is within the 25 minute intervals or not.
        if int(date) < 20210329:  # -> 08 to 16.70
            if 82500 < int(time) < 160500:
                cls.high_activity_flag = 0

        elif int(date) >= 20210329:  # -> 07 to 17.30
            if 72500 < int(time) < 160500:
                cls.high_activity_flag = 0

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