#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config dict generator."""

import json
import copy

PATH = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/" \
       "reinforcement_learning/base_configs/"


def create_base_config_file(config_dict, name, path=PATH):
    """
    Takes config dict as input, cleans it from instacnes and stores
    it to base_dict path.
    """

    # Copy config and remove the instances.
    base_config = copy.deepcopy(config_dict)
    # Remove Env.
    base_config['env'] = 0
    # Remove Replay.
    base_config['env_config']['config']['replay'] = 0
    # pathname.
    pathname = path + name + "_base_config" + ".json"
    with open(pathname, 'w') as fp:
        json.dump(base_config, fp)