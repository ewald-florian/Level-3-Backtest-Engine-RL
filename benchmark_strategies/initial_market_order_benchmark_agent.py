#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Strategy: It submits a market order of the total inventory
in the first step of the episode. I compute the resulting implementation
shortfall as benchmark for optimal execution strategies.
"""

import numpy as np
import pandas as pd

from market.market_interface import MarketInterface
from market.market import Market
from replay_episode.replay import Replay
from agent.agent_metrics import AgentMetrics
from agent.agent_order import OrderManagementSystem as OMS
from agent.agent_trade import AgentTrade
from context.context import Context


class InitialMarketOrderTrader:
    """
    Test Backtest Engine with random submissions and cancellations.
    """
    def __init__(self,
                 initial_inventory: int = 100_0000,
                 verbose=True):

        self.initial_inventory = initial_inventory
        self.verbose = verbose

        self.agent_metrics = AgentMetrics()
        self.market_interface = MarketInterface()
        self.first_step_over = False

    def submit_initial_market_order(self):
        """Submit a market sell order with the entire remaining inventory"""
        best_bid = Market.instances["ID"].best_bid
        ticksize = Market.instances["ID"].ticksize
        marketable_limit = best_bid - 1000*ticksize

        # Place order via market interface.
        if not self.first_step_over:
            self.market_interface.submit_order(side=2,
                                               limit=marketable_limit,
                                               quantity=self.initial_inventory)
            if self.verbose:
                print(f'(RL AGENT) Submission: limit: {marketable_limit}  '
                      f'qt: {self.initial_inventory}')

        # Set first_step_over flag to True:
        self.first_step_over = True

    def reset(self):
        pass


# Run backtest.
if __name__ == '__main__':

    # inst: -> generates episode start list
    # TODO: I HAVE TO USE THE EXACT SAME SAMPLING SET-UP AS IN THE RL-STRATEGY
    #  TO GET THE SAME EPISODE LIST (or better I even save the episode list)
    replay = Replay(identifier="BAY",
                    start_date="2021-01-01",
                    end_date="2021-04-30",
                    episode_length="10s",
                    frequency="1m",
                    seed=42,
                    shuffle=False,
                    random_identifier=False,
                    exclude_high_activity_time=False,
                    )
    # print(replay.episode_start_list)

    # Note: Use base-reset instead of RL reset.
    # Episode is counted automatically inside episode meaning, everytime
    # base reset is called, the next episode in episode_start_list is
    # created.
    results = []
    # TODO: Problem, I dont know how many of the episodes in episode
    #  start list can actually be loaded successfulle -> what should I
    #  iterate over? (zur not try except
    for episode in range(10):
        print("episode number:", episode)

        # Build new episode and reset env.
        replay.base_reset()
        print("episode_start:", replay.episode.episode_start)

        # Initialize agent in each new episode.
        agent = InitialMarketOrderTrader()

        print("Episode Len: ", replay.episode.__len__())

        # Iterate over Episode Steps.
        for i in range(replay.episode.__len__()):

            # Call normal step.
            replay.normal_step()

            # Apply stragey.
            agent.submit_initial_market_order()

            # The episode ends early when the entire inventory is sold.
            if (agent.agent_metrics.executed_quantity >=
                agent.initial_inventory):
                # Break terminates the loop over the steps and goes to the next
                # episode.
                results.append(np.array([replay.episode.episode_start,
                                         agent.agent_metrics.overall_is(
                                             scaling_factor=1)]))
                print(pd.DataFrame(results))
                break

# Store results to DF:
# TODO: name vom asset und strategie in filename einbauen.
path = '/Users/florianewald/Desktop/test_initial_market.csv'
df = pd.DataFrame(results)
df.to_csv(path, index=False)


# TODO: Sicherstellen, dass Market Impact aktiviert ist (wobei es ja eh nur
#  einen Zeitschritt dauern sollte.
# TODO: Compute IS and store together with Episode start as Identifier
