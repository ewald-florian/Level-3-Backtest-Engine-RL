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

from utils.test_result_path import generate_test_result_path
from utils.initial_inventory import initial_inventory_dict


class TWAPTrader:
    """
    TWAP:

    - Split initial volume into N chunks and submit them evenly distributed
      over the episode as market orders.

    - Forced execution before the end of the episode.

    """

    def __init__(self,
                 episode_start,
                 initial_inventory=None,
                 num_twap_orders: int = 4,
                 episode_len: str = "10s",
                 verbose: bool = True):
        """
        When initialized, the agent generates a TWAP execution plan
        including the unix timestamps at which the TWAP orders will be
        submitted.
        """

        self.initial_inventory = initial_inventory
        self.verbose = verbose

        # -- Create execution plan.
        # Convert pd timestamp to unix time.
        episode_start_unix = (episode_start - pd.Timestamp(
            "1970-01-01", tz='UTC')) // pd.Timedelta('1ns')

        self.twap_quantity = round(initial_inventory / num_twap_orders)
        episode_len_ns = pd.Timedelta(episode_len).delta
        twap_interval_unix = episode_len_ns / num_twap_orders

        # The first execution is scheduled immediately after episode start.
        self.twap_execution_plan_unix = [episode_start_unix]
        # Add the other executions times in unix time.
        for step in range(1, num_twap_orders):
            self.twap_execution_plan_unix.append(episode_start_unix +
                                                 step * twap_interval_unix)

        # Add zero after last submission to avoid more submissions.
        self.twap_execution_plan_unix.append(0)

        # -- compositions
        self.agent_metrics = AgentMetrics()
        self.market_interface = MarketInterface()

        # -- variables
        self.submission_counter = 0
        self.submit_next_order = False
        self.forced_execution_done = False

    def twap_step(self):
        """
        Submit a market order, if a new twap interval is reached.
        """
        # Set submit next order flag True when new twap step is reached.
        next_sub_unix = self.twap_execution_plan_unix[self.submission_counter]
        if next_sub_unix > Market.instances["ID"].timestamp:
            self.submit_next_order = True

        # Submit TWAP order if flag is True.
        if self.submit_next_order == True:

            ticksize = Market.instances["ID"].ticksize
            best_bid = Market.instances["ID"].best_bid
            marketable_limit = best_bid - 1000 * ticksize

            # Place TWAP order via market interface.
            self.market_interface.submit_order(side=2,
                                               limit=marketable_limit,
                                               quantity=round(
                                                   self.twap_quantity, -4))
            if self.verbose:
                print(f'(RL AGENT) Submission: limit: {marketable_limit}  '
                      f'qt: {round(self.twap_quantity, -4)}')

            # Count the submission:
            self.submission_counter += 1
            # Set flag False until next twap step is reached
            self.submit_next_order = False

        else:
            pass

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
            # Note: I don't need to count the forced submission.

    def reset(self):
        pass


# Run backtest.
if __name__ == '__main__':

    # TODO: I can also iterate over the symbols to let it run in one loop.
    # SET UP THE BACKTEST:
    # ---------------------
    name = "twap_Avg-10s-Vol_"
    symbol = "BMW"
    # TODO: Use same initial inventory as for RL-agent.
    initial_inv = initial_inventory_dict[symbol]['Avg-10s-Vol'] * 1_0000
    testset_start = "2021-05-14" #"2021-01-01" # "2021-05-14"
    testset_end = "2021-06-30" #"2021-04-30", "2021-06-30"
    episode_len = "10s"
    frequency = "5m"
    num_iterations_to_store_results = 1_000
    # ----------------------

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
    result_path = generate_test_result_path(symbol=replay.identifier,
                                            benchmark_strategy=name)

    # Iterate over episodes.
    for episode in range(1_000_000):  # -> end will not be reached.

        # Try as long as there are remaining episodes.
        try:

            # Build new episode and reset env.
            replay.base_reset()
            episode_start = replay.episode.episode_start

            # Initialize agent in each new episode.
            agent = TWAPTrader(episode_start=episode_start,
                               initial_inventory=initial_inv,
                               num_twap_orders=4,
                               episode_len=episode_len,
                               verbose=False)

            # Iterate over Episode Steps.
            for i in range(replay.episode.__len__()):

                # Call normal step.
                replay.normal_step()

                # Apply strategy.
                agent.twap_step()

                # Forced liquidation 2 steps before episode end.
                if (replay.episode._step == (replay.episode.__len__() - 5) and \
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

                    break

        # Make exception since the episode start list will be over.
        except:
            pass

    # Store final  results to DF:
    df = pd.DataFrame(results, columns=["episode_start_time",
                                        "is", "vwap_sell", "episode_len"])

    df.to_csv(result_path, index=False)
    print(df)
