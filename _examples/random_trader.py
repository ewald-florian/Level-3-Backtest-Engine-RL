"""For Demonstration Purposes"""

from market.market_interface import MarketInterface
from market.market import Market
from replay.replay import Replay
from agent.agent_metrics import AgentMetrics
from agent.agent_order import OrderManagementSystem as OMS

import numpy as np


class RandomTrader:
    """
    Test Backtest Engine with random submissions and cancellations.
    """

    def __init__(self,
                 number_range: int = 1000):

        self.range = number_range
        self.metrics = AgentMetrics()
        self.mi = MarketInterface()

    def submit_random_orders(self):

        lucky_number = np.random.randint(0, self.range)

        # buy order
        if lucky_number == 42:
            # submit order via market interface
            best_ask = Market.instances['ID'].best_ask

            self.mi.submit_order(side=1,
                                 limit=best_ask,
                                 quantity=lucky_number * 1e4)

        # sell order
        if lucky_number == 24:
            best_bid = Market.instances['ID'].best_ask

            self.mi.submit_order(side=2,
                                 limit=best_bid,
                                 quantity=lucky_number * 1e4)

        # cancellation
        if lucky_number == 69:

            # filter OMS for active orders
            l = list(filter(lambda d: d['template_id'
                                      ] == 99999, OMS.order_list))
            if l:
                # get message_id of first active order
                message_id = l[0]['message_id']

                # cancel order
                self.mi.cancel_order(order_message_id=message_id)
            else:
                pass

        if lucky_number == 96:
            # log AgentMetrics
            print(self.metrics)


if __name__ == '__main__':

    # inst: -> generate episode start list
    replay = Replay(identifier="FME",
                    start_date="2022-02-16",
                    end_date="2022-02-16",
                    episode_length="1H",
                    frequency="1m",
                    seed=42,
                    shuffle=True,
                    random_identifier=False,
                    exclude_high_activity_time=True,
                    )

    replay.rl_reset()  # -> build new episode

    randomtrader = RandomTrader()

    print("Episode Len: ", replay.episode.__len__())

    for i in range(replay.episode.__len__()):
        replay.normal_step()
        randomtrader.submit_random_orders()
