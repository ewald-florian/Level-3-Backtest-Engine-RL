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

from utils.benchmark_result_path import generate_benchmark_result_path
from utils.initial_inventory import initial_inventory_dict


class SubmitLeaveTrader:
    """
    Submit and Leave Trader:

    - Submits a limit order with the entire inventory in the beginning of the
        episode, e.g. with best-ask as limit.

    - Forced execution before the end of the episode.
    """

    def __init__(self,
                 initial_inventory=None,
                 verbose=True):

        self.initial_inventory = initial_inventory
        self.verbose = verbose

        self.agent_metrics = AgentMetrics()
        self.market_interface = MarketInterface()
        self.first_step_over = False
        self.forced_execution_done = False

    def submit_initial_limit_order(self):
        """Submit a limit sell order with the entire remaining inventory"""

        best_ask = Market.instances["ID"].best_ask

        # Place order via market interface.
        if not self.first_step_over:
            self.market_interface.submit_order(side=2,
                                               limit=best_ask,
                                               quantity=self.initial_inventory)
            if self.verbose:
                print(f'(RL AGENT) Submission: limit: {best_ask}  '
                      f'qt: {self.initial_inventory}')

        # Set first_step_over flag to True:
        self.first_step_over = True

    def force_execution_episode_end(self):
        """
        Submit a market sell order with the entire remaining inventory.
        Should be triggered, before episode ends.
        """
        ticksize = Market.instances["ID"].ticksize
        best_bid = Market.instances["ID"].best_bid
        marketable_limit = best_bid - 1000 * ticksize

        remaining_inv = (self.initial_inventory -
                         self.agent_metrics.executed_quantity)

        # Place order via market interface.
        if not self.forced_execution_done:
            self.market_interface.submit_order(side=2,
                                               limit=marketable_limit,
                                               quantity=remaining_inv)
            if self.verbose:
                print(f'(RL AGENT) Forced Sub: limit: {marketable_limit}  '
                      f'qt: {remaining_inv}')

            # Set forced execution done flag.
            self.forced_execution_done = True

    def reset(self):
        pass


# Run backtest.
if __name__ == '__main__':

    # TODO: I can also iterate over the symbols to let it run in one loop.
    # SET UP THE BACKTEST:
    # -------------------------
    name = "submit_and_leave_Avg-10s-Vol_"
    symbol = "DTE"
    # TODO: Use same initial inventory as for RL-agent.
    initial_inv = initial_inventory_dict[symbol]['Avg-10s-Vol'] * 1_0000
    testset_start = "2021-05-14" #"2021-01-01" # "2021-05-14"
    testset_end = "2021-06-30" #"2021-04-30", "2021-06-30"
    episode_len = "10s"
    frequency = "5m"
    num_iterations_to_store_results = 1_000

    # -------------------------

    # inst: -> generates episode start list
    # TODO: I HAVE TO USE THE EXACT SAME SAMPLING SET-UP AS IN THE RL-STRATEGY
    #  TO GET THE SAME EPISODE LIST (or better I even save the episode list)
    replay = Replay(identifier=symbol,
                    start_date=testset_start,  # "2021-05-14",# "2021-01-01",
                    end_date=testset_end,  # "2021-06-30",#"2021-04-30",
                    episode_length=episode_len,
                    frequency=frequency,
                    seed=42,
                    shuffle=False,
                    random_identifier=False,
                    exclude_high_activity_time=False,
                    verbose=False
                    )
    # print(replay.episode_start_list)

    results = []
    result_path = generate_benchmark_result_path(symbol=replay.identifier,
                                                 benchmark_strategy=name)

    for episode in range(1_000_000):  # -> end will not be reached.

        # Try as long as there are remaining episodes.
        try:

            # Build new episode and reset env.
            replay.base_reset()

            # Initialize agent in each new episode.
            agent = SubmitLeaveTrader(initial_inventory=initial_inv,
                                      verbose=False)

            # Iterate over Episode Steps.
            for i in range(replay.episode.__len__()):


                # Call normal step.
                replay.normal_step()

                # Apply strategy.
                agent.submit_initial_limit_order()


                # Forced liquidation 2 steps before episode end.
                if (replay.episode._step == (replay.episode.__len__() - 5) and
                        not agent.forced_execution_done):

                    agent.force_execution_episode_end()

                # The episode ends early when the entire inventory is sold.
                if (agent.agent_metrics.executed_quantity >=
                        agent.initial_inventory):
                    # Break terminates the loop over the steps and goes to the next
                    # episode.
                    results.append(np.array([replay.episode.episode_start,
                                             agent.agent_metrics.overall_is(
                                                 scaling_factor=1),
                                             agent.agent_metrics.vwap_sell,
                                             replay.episode._step]))

                    # Intermediate result storage (will be overwritten)
                    if episode % num_iterations_to_store_results == 0:
                        df = pd.DataFrame(results,
                                          columns=["episode_start_time",
                                                   "is", "vwap_sell",
                                                   "episode_len"])
                        df.to_csv(result_path, index=False)
                        # Print to terminal.
                        print(df)
                    # Break Loop After DF was saved.
                    break

        # Make exception since the episode start list will be over.
        except:
            pass

    # Store results to DF:
    df = pd.DataFrame(results, columns=["episode_start_time",
                                        "is", "vwap_sell", "episode_len"])

    print(df)
    df.to_csv(result_path, index=False)
