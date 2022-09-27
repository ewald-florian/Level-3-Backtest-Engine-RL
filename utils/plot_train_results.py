#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Plot training results. """
__author__ = "florian"
__date__ = "2022-09-25"
__version__ = "0.1"

# TODO: testen

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12,8)


def plot_train_results(result_path: str):
    """
    Plot training results.
    :param result_path,
        str, path to train result csv file.
    """

    df = pd.read_csv(result_path)
    df.dropna()
    df.plot(x="n", y=["episode_reward_mean", "episode_reward_min",
                      "episode_reward_max"], secondary_y=True)

    plt.show()
