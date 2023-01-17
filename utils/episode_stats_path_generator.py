#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
"""Generate Unique Paths to store Epsiode Stats of Experiments in"""

__author__ = "florian"

import time
import json
import platform

local_dir = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/" \
            "reinforcement_learning/episode_stats/"
server_dir = "/home/jovyan/Level-3-Backtest-Engine-RL/" \
             "reinforcement_learning/episode_stats/"

if platform.system() == 'Darwin':  # macos
    default_base_dir = local_dir
elif platform.system() == 'Linux':
    default_base_dir = server_dir
else:
    raise Exception('Specify base_dir in utils/result_path_generator.py')


def generate_episode_stats_path(name: str,
                                base_dir: str = default_base_dir) -> str:
    """
    Generate unique episode stats path names (including filename) with the
    current time and the name of the experiment. Name can be used to indicate
    specific attributes of the experiment.

    This function does not only generate a file name but it also creates a
    json file filled with an empty list. This list can later be used to
    append the statistics.

    :param name,
        str, name of the experiment
    :param base_dir
        str, directory where results should be stored.
    :return result_path
        str, unique path to store result
    """

    time_str = time.strftime("_%Y%m%d_%H%M%S")
    stats_path = base_dir + name + time_str + "_episode_stats.json"

    # Serialize Empty list which will later be appended with dictionaries.
    json_object = json.dumps([], indent=4)

    # Store empty list into the json file.
    with open(stats_path, "w") as outfile:
        outfile.write(json_object)

    return stats_path


stats_path = generate_episode_stats_path(name="inthematrix")
print(stats_path)