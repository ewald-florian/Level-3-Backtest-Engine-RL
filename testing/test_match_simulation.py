"""For Demonstration Purposes"""

from market.market_interface import MarketInterface
from market.market import Market
from replay_episode.replay import Replay
from agent.agent_metrics import AgentMetrics
from agent.agent_order import OrderManagementSystem as OMS

import numpy as np


class Trader:
    """
    Test match_simulation.
    """

    def __init__(self):

        self.counter = 0
        self.metrics = AgentMetrics()
        self.mi = MarketInterface()

    def trade(self):

        # -- fill OMS with limit orders
        # buy order
        if self.counter < 10:
            # submit order via market interface
            best_bid = Market.instances['ID'].best_bid

            self.mi.submit_order(side=1,
                                 limit=best_bid,
                                 quantity=333_0000)

        if 10 < self.counter < 15:
            # submit order via market interface
            best_ask = Market.instances['ID'].best_ask

            self.mi.submit_order(side=2,
                                 limit=best_ask,
                                 quantity=333_0000)

        # -- submit markatable orders
        if self.counter == 20:
            # submit order via market interface
            best_ask = Market.instances['ID'].best_ask

            self.mi.submit_order(side=1,
                                 limit=best_ask,
                                 quantity=222_0000)

        if self.counter == 30:
            # submit order via market interface
            best_bid = Market.instances['ID'].best_bid

            self.mi.submit_order(side=2,
                                 limit=best_bid,
                                 quantity=222_0000)

        self.counter += 1


if __name__ == '__main__':

    # inst: -> generate episode start list
    replay = Replay(identifier="FME",
                    start_date="2022-02-16",
                    end_date="2022-02-16",
                    episode_length="1H",
                    frequency="1m",
                    seed=40,
                    shuffle=True,
                    random_identifier=False,
                    exclude_high_activity_time=True,
                    )

    replay.rl_reset()  # -> build new episode

    trader = Trader()

    print("Episode Len: ", replay.episode.__len__())

    for i in range(33):#replay.episode.__len__()):
        print(i)
        replay.normal_step()
        trader.trade()