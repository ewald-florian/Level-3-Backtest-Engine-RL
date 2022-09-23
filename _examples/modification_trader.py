"""For Demonstration Purposes"""

from market.market_interface import MarketInterface
from market.market import Market
from replay.replay import Replay
from agent.agent_trade import AgentTrade
from agent.agent_metrics import AgentMetrics
from agent.agent_order import OrderManagementSystem as OMS



class ModificationTrader:

    """
    Test Backtest Engine.
    """

    def __init__(self, number_range:int=100):

        self.range = number_range
        self.metrics = AgentMetrics()
        self.agent_trade = AgentTrade.history
        self.mi = MarketInterface()
        self.counter = 0


    def apply_strategy(self):

        self.counter = self.counter + 1

        # 1) order modification with new priority time

        # -- submission

        if self.counter == 10:

            # submit passive order
            midpoint = Market.instances['ID'].midpoint

            ticksize = Market.instances['ID'].ticksize

            passive_limit = midpoint - (2*ticksize)

            self.mi.submit_order(side=1,
                            limit=passive_limit,
                            quantity=77_0000)

            # check OMS
            print(OMS.order_list)

        # -- modification

        if self.counter == 100:

            # check if the limit order is still active (not yet executed)
            if OMS.order_list[-1]['template_id'] == 99999:

                # get message_id of order to modify
                mod_id = OMS.order_list[-1]['message_id']

                # use best_ask as new aggressive limit
                aggressive_limit = Market.instances['ID'].best_ask

                # decrease quantity due to the worse limit
                new_quantity = 30_0000

                # note: order modification will affect priority time!

                # modify order
                self.mi.modify_order(order_message_id=mod_id,
                                     new_price=aggressive_limit,
                                     new_quantity=new_quantity)

                # check OMS
                print("ORDER LIST", OMS.order_list)
                # check trade list
                print("Trade List", self.agent_trade)

        # 1) order modification with same priority time

        # -- submit
        if self.counter == 150:

            # submit new passive order
            midpoint = Market.instances['ID'].midpoint

            ticksize = Market.instances['ID'].ticksize

            passive_limit = midpoint - (2 * ticksize)

            self.mi.submit_order(side=1,
                                 limit=passive_limit,
                                 quantity=77_0000)

            # check OMS
            print(OMS.order_list)

        # -- modify
        if self.counter == 200:

            # check if the limit order is still active (not yet executed)
            if OMS.order_list[-1]['template_id'] == 99999:

                # get message_id of order to modify
                mod_id = OMS.order_list[-1]['message_id']

                # decrease quantity
                new_quantity = 30_0000

                # note: order modification will affect priority time!

                # modify order
                self.mi.modify_order(order_message_id=mod_id,
                                     new_quantity=new_quantity)

                # check OMS
                print("ORDER LIST", OMS.order_list)


if __name__ == '__main__':

    # inst: -> generate episode start list
    replay = Replay(identifier="FME",
                 start_date="2022-02-16",
                 end_date="2022-02-16",
                 episode_length = "1H",
                 frequency ="1m",
                 seed = 42,
                 shuffle=True,
                 random_identifier=False,
                 exclude_high_activity_time=True,
        )

    replay.reset() # -> build new episode

    moditrader = ModificationTrader()

    print("Episode Len: ", replay.episode.__len__())

    for i in range(210): #replay.episode.__len__()):

        replay.normal_step()
        moditrader.apply_strategy()

        if i%100==0:
            # log market
            print('Market:', Market.instances['ID'].state_l1)


