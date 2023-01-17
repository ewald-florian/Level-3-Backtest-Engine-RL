#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class to manage the episode statistics"""

import json


class EpisodeStats:
    """
    Class attribute path_name is used to store the path to the json file
    to store episode stats during the current training loop.
    """
    # class attribute tuple
    path_name = None

    @classmethod
    def __init__(cls,
                 path_name):
        """
        Stores path name.
        :param path_name
            str, path name to json file
        """
        cls.path_name = path_name

    @classmethod
    def store_episode_results(cls,
                              oms,
                              agent_trade):
        """Store episode statistics to json file."""

        # Combine OMS and AgentTrade to dict.
        episode_stat_dict = {}
        episode_stat_dict["OMS"] = oms
        episode_stat_dict["AT"] = agent_trade

        # Read the list from json file.
        with open(cls.path_name) as fp:
            stats_list = json.load(fp)

        # Append dict with new stats to list.
        stats_list.append(episode_stat_dict)

        # Write appended list to the original json file.
        with open(EpisodeStats.path_name, 'w') as json_file:
            json.dump(stats_list, json_file)
