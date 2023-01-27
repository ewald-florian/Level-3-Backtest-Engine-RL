#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""
TWAP Agent:

input:
- Parent Order Size
- Nubmer of Steps
derived:
- Child Order Size
- Submission times

required:
- The time delta of the episode needs to be known in advance (either directly
 taking episode end from the build episode or assuming that the demanded
 episode length is actually given, e.g. if I tell replay 1m, I assume the
 episode actually has 1m
- Compute the trading frequency needed to sell the Parent orders within the
 number of steps
- Take special care for the episode end, is everything sold? did I place a
 final order which cannot even be filled?


Discussion:
- Should the first trade be directly placed in the first step?
- Not all orders are executed (this problem would not exist in reality since
 orders can be placed anytime...)

Alternative Ideas:
- The number of child orders could be computed internally depending on the
episode length
- I could work with nanoseconds instead of pd datetime (number of nanoseconds
of the time horizon, nanosecond time of first timestamp of the episode)
- I could create a pre-schedule with timestamps at which orders must be
 submitted instead of doing it on the fly with "last_submission_time"
"""

import pandas as pd


from market.market_interface import MarketInterface
from market.market import Market
from replay_episode.replay import Replay


class TWAPAgent:

    def __init__(self,
                 parent_order_size,
                 number_of_child_orders,
                 trading_horizon,
                 buffer_factor=0.5):

        self.num_childorders = number_of_child_orders
        self.child_order_size = parent_order_size / number_of_child_orders
        trading_horizon = pd.Timedelta(trading_horizon)*buffer_factor
        self.trading_interval = trading_horizon / number_of_child_orders

        # flag to place the first trade
        self.first_trade = True
        self.last_submission_time = None
        self.order_counter = 0

        #self.metrics = AgentMetrics()
        #self.agent_trade = AgentTrade.history
        self.mi = MarketInterface()

    def apply_strategy(self):

        # get current timestamp from Market
        current_time = Market.instances['ID'].timestamp_datetime

        if self.first_trade:
            #DEBUGGING
            print(self.trading_interval)
            print(self.child_order_size)

            # directly place first order
            best_ask = Market.instances['ID'].best_ask
            self.mi.submit_order(side=2,
                                 limit=best_ask,
                                 quantity=self.child_order_size)
            # track submission
            self.last_submission_time = current_time
            self.order_counter += 1
            print(f"order {self.order_counter} placed")
            # set first trade flag False
            self.first_trade = False

        elif current_time - self.last_submission_time > self.trading_interval \
                and self.order_counter < self.num_childorders:
            print("current time", current_time)
            print("last sub time", self.last_submission_time)
            print("diff", current_time - self.last_submission_time)
            print("interval", self.trading_interval)
            # submit order
            best_ask = Market.instances['ID'].best_ask
            self.mi.submit_order(side=2,
                                 limit=best_ask,
                                 quantity=self.child_order_size)
            # update last_submission_time
            self.last_submission_time = current_time
            self.order_counter += 1
            print(f"order {self.order_counter} placed")

        else:
            pass


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

    trader = TWAPAgent(parent_order_size=1000,
                       number_of_child_orders=20,
                       trading_horizon="1m")

    print("Episode Len: ", replay.episode.__len__())

    for i in range(replay.episode.__len__()):
        replay.normal_step()
        trader.apply_strategy()
        print(Market.instances['ID'].timestamp_datetime)


# E
# 2022-02-16 10:50:58.587303462
# 2022-02-16 10:51:58.013558939








