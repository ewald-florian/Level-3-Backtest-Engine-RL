#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
"""Generate Unique Paths to store Results of Experiments"""

import time
import platform
import os

local_dir = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/" \
            "reinforcement_learning/train_results/"
local_2 = "/Users/candis/Desktop/Level-3-Backtest-Engine-RL/" \
            "reinforcement_learning/train_results/"
server_dir_old = "/home/jovyan/Level-3-Backtest-Engine-RL/" \
             "reinforcement_learning/train_results/"
server_dir = "/home/jovyan/train_results/"

if platform.system() == 'Darwin':  # macos
    if os.path.isdir(local_2):
        default_base_dir = local_2
    else:
        default_base_dir = local_dir
elif platform.system() == 'Linux':
    default_base_dir = server_dir
else:
    raise Exception('Specify base_dir in utils/result_path_generator.py')


def generate_result_path(name: str, base_dir: str = default_base_dir) -> str:
    """
    Generate unique result path names (including filename) with the current
    time and the name of the experiment. Name can be used to indicate specific
    attributes of the experiment.
    :param name,
        str, name of the experiment
    :param base_dir
        str, directory where results should be stored.
    :return result_path
        str, unique path to store result
    """

    time_str = time.strftime("_%Y%m%d_%H%M%S")
    result_path = base_dir + name + time_str + "_results.csv"

    return result_path
