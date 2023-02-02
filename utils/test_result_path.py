#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Result Paths for benchmark strategies. Can also be used for
normal strategies.
"""
import time
import platform
import os

'''
# benchmark paths:
local_dir = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/" \
            "benchmark_strategies/benchmark_results/"
server_dir = "/home/jovyan/benchmark_results/"
'''
# test paths:
local_dir = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/" \
            "test_results/"
local_2 = "/Users/candis/Desktop/Level-3-Backtest-Engine-RL/test_results/"
server_dir = "/home/jovyan/test_results/"

if platform.system() == 'Darwin':  # macos
    if os.path.isdir(local_2):
        default_base_dir = local_2
    else:
        default_base_dir = local_dir
elif platform.system() == 'Linux':
    default_base_dir = server_dir
else:
    raise Exception('Specify base_dir in utils/result_path_generator.py')


def generate_test_result_path(symbol,
                              strategy_name,
                              base_path=default_base_dir):
    """
    Generates a result path with symbol and strategy name,
    e.g .../FME_submit_and_leave.cvs
    """
    time_str = time.strftime("_%Y%m%d_%H%M%S")
    result_path = base_path+time_str+"-"+symbol+"-"+strategy_name+".cvs"

    return result_path