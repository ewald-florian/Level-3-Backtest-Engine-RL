#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
Test how to handle remaining time and inventory, how to normalize etc.
This is a research-agent in prepararion of the RL agents.

Note:
- This is a "timestamp" based version
- The Endtime is estimated in the first iteration can slightly vary from actual
- Both remaining and elapsed time work and can be normalised
- Remaining inventory works and can be normalised

"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-11-10"
__version__ = "0.1"
# ----------------------------------------------------------------------------

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
                 episode_length,
                 parent_order_size): #_0000 (4 extra digits!)

        self.initial_inventory = parent_order_size
        # TODO: where can I get start and end time?
        #  can I access episode? It would not make sense to have it
        #  as hardcoded input since the episodes will be sampled automatically
        #  later... maybe I can store it in some class attribute to access it
        #  from everywhere... like Episode.parameters = list() The problem is,
        #  that agent is actually instatiated befrore replay and hence before
        #  the episode!
        #  One way that should always work is to take the first market timestamp
        #  as starting time and then assume that the end time will be start
        #  time plus time interval. WARNING: this can lead to slight deviation
        #  which may cause the normalizaiton to be outside the [0,1] range in
        #  the end of an episode... I could clip the normed value e.g. if
        #  current_time > end_time, norm = 01
        #
        self.start_time = None
        self.end_time = None
        # time delta
        self.time_delta = pd.to_timedelta(episode_length)
        self.mi = MarketInterface()

        # dynamic attributes
        self.first_step = True
        self.remaining_inventory = parent_order_size
        self.number_of_trades = 0

    def apply_strategy(self):

        # -- TIME

        # get current timestamp from Market
        current_time = Market.instances['ID'].timestamp_datetime

        # define start and end-time in the first episode
        if self.first_step:
            self.start_time = current_time
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


        # -- INVENTORY
        # remaining_inventory has to be decreased every time a trade happens

        # NOTE: there can be several new trades at the same time, I cannot
        #  only look at the one last trade!
        new_trades = len(AgentTrade.history) - self.number_of_trades

        # update remaining inventory and count trades in number_of_trades
        if new_trades:
            for trade in AgentTrade.history[-new_trades:]:
                quantity = trade['executed_volume']
                self.remaining_inventory -= quantity
                self.number_of_trades += 1

        remaining_inventory_norm = (self.remaining_inventory /
                                    self.initial_inventory)

        print("remaininng inv: ", self.remaining_inventory)
        print("initial inv ", self.initial_inventory)

        print("remaining_inventory_nomr", remaining_inventory_norm)

        # some trading
        number = np.random.randint(20)
        if number == 1:
            self.mi.submit_order(side=2,
                            quantity=40_0000,
                            limit=Market.instances['ID'].best_ask)


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

    # In this case, episode is already known but this is not the case in RL loop
    trader = TimeInventoryAgent(episode_length="1m",
                                parent_order_size=1000_0000)

    print("Episode Len: ", replay.episode.__len__())

    for i in range(replay.episode.__len__()):
        replay.normal_step()
        trader.apply_strategy()
        print(Market.instances['ID'].timestamp_datetime)