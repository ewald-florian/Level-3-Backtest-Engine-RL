#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ---------------------------------------------------------------------------
"""Generate Unique Paths to store Results of Experiments in"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-09-25"
__version__ = "0.1"
# ----------------------------------------------------------------------------
import time
import platform

local_dir = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/reinforcement_learning/training_results/"
server_dir = "/home/jovyan/_shared_storage/temp/A7_eobi_data/downloader_working_05/"

if platform.system() == 'Darwin':  # macos
    default_base_dir = local_dir
elif platform.system() == 'Linux':
    default_base_dir = server_dir
else:
    raise Exception('Specify base_dir in result_path_generator.py')


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
