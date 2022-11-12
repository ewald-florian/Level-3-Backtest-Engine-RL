#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Test how to handle remaining time, using Unix time (nuber of nanoseconds)

Note:
- This is a unix time based version
- Should be more efficient/faster than the timestamp version

"""

__author__ = "florian"
__date__ = "2022-11-10"
__version__ = "0.1"

import pandas as pd
import numpy as np

from market.market import Market
from market.market_interface import MarketInterface
from agent.agent_trade import AgentTrade
from replay_episode.replay import Replay

# TODO:
#  time as pd datetime (better for debugging)
#  time as nanoseconds (should be much more efficient!)
#  - should later be part of ObservationSpace (if possible)


class TimeInventoryAgent:

    def __init__(self,
                 episode_length="1m"): #_0000 (4 extra digits!)

        self.start_time = None
        self.end_time = None
        # time delta in NANOSECONDS (with .delta)
        self.time_delta = pd.to_timedelta(episode_length).delta
        self.first_step = True

        self.mi = MarketInterface()

    def apply_strategy(self):

        # -- TIME

        # get current timestamp from Market (in UNIX)
        current_time = Market.instances['ID'].timestamp

        # define start and end-time in the first episode
        if self.first_step:
            self.start_time = current_time
            # Add nanoseconds delta to start time to get end time
            self.end_time = current_time + self.time_delta
            self.first_step = False
            print("start_time: ", self.start_time)
            print("end_time: ", self.end_time)

        # elapsed time:
        elapsed_time = current_time - self.start_time
        elapsed_time_normed = elapsed_time / self.time_delta
        print("elapsed_time_normed", elapsed_time_normed)
        # remaining time:
        remaining_time = self.end_time - current_time
        remaining_time_normed = remaining_time / self.time_delta
        print("remaining_time_normed", remaining_time_normed)


if __name__ == '__main__':

    # inst: -> generate episode start list
    replay = Replay(identifier="FME",
                    start_date="2022-02-16",
                    end_date="2022-02-16",
                    episode_length="1m",
                    frequency="1m",
                    seed=42,
                    shuffle=True,
                    random_identifier=False,
                    exclude_high_activity_time=True,
                    )

    replay.base_reset()  # -> build new episode

    # In this case, episode is already known but this is not the case in RLloop
    trader = TimeInventoryAgent(episode_length="1m")

    print("Episode Len: ", replay.episode.__len__())

    for i in range(replay.episode.__len__()):
        replay.normal_step()
        trader.apply_strategy()
        print(Market.instances['ID'].timestamp_datetime)
