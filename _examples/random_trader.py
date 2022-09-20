"""For Demonstration Purposes"""

from market.market_interface import MarketInterface as MI
from market.market import Market
from replay.replay import Replay
from agent.agent_metrics import AgentMetrics
from agent.agent_order import OrderManagementSystem as OMS

import numpy as np


class RandomTrader:

    def __init__(self, number_range:int=10_000):

        self.range = number_range
        self.metrics = AgentMetrics()


    def submit_random_orders(self):

        lucky_number = np.random.randint(0, self.range)

        # buy order
        if lucky_number == 42:

            # submit order via market interface
            best_ask = Market.instances['ID'].best_ask

            MI.submit_order(side=1,
                            limit=best_ask,
                            quantity=lucky_number*1e4)

        # sell order
        if lucky_number == 24:

            best_bid = Market.instances['ID'].best_ask

            MI.submit_order(side=2,
                            limit=best_bid,
                            quantity=lucky_number * 1e4)

        # cancellation
        if lucky_number == 69:

            # filter OMS for active orders
            l = list(filter(lambda d: d['template_id'
                                      ] == 99999, OMS.order_list))
            if l:
                # get message_id of first active order
                id = l[0]['message_id']

                # cancel order
                MI.cancel_order(order_message_id=id)
            else:
                pass

        if lucky_number == 96:
            # log Position Value from AgentMetrics
            print(">>> POSITION VALUE: ", round(
                self.metrics.position_value/1e-8),2)


if __name__ == '__main__':

    # inst: -> generate episode start list
    replay = Replay(identifier="FME",
                 start_date="2022-02-16",
                 end_date="2022-02-16",
                 episode_length = "10m",
                 frequency ="1m",
                 seed = 42,
                 shuffle=True,
                 random_identifier=False,
                 exclude_high_activity_time=True,
        )

    replay.reset() # -> build new episode
    randomtrader = RandomTrader()
    print("Episode Len: ", replay.episode.__len__())

    for i in range(replay.episode.__len__()):

        replay.step()
        randomtrader.submit_random_orders()

        if i%100==0:
            print('Market:',Market.instances['ID'].state_l1)









