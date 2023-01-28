#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Result Paths for benchmark strategies. Can also be used for
normal strategies.
"""
import time
import platform


local_dir = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/" \
            "benchmark_strategies/benchmark_results/"
server_dir = "/home/jovyan/benchmark_results/"

if platform.system() == 'Darwin':  # macos
    default_base_dir = local_dir
elif platform.system() == 'Linux':
    default_base_dir = server_dir
else:
    raise Exception('Specify base_dir in utils/result_path_generator.py')


def generate_benchmark_result_path(symbol,
                                   benchmark_strategy,
                                   base_path=default_base_dir):
    """
    Generates a result path with symbol and strategy name,
    e.g .../FME_submit_and_leave.cvs
    """
    time_str = time.strftime("_%Y%m%d_%H%M%S")
    result_path = base_path+time_str+"-"+symbol+"-"+benchmark_strategy+".cvs"

    return result_path