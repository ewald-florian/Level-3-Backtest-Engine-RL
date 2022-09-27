"""For Demonstration Purposes"""

from market.market_interface import MarketInterface
from market.market import Market
from replay_episode.replay import Replay
from agent.agent_trade import AgentTrade
from agent.agent_metrics import AgentMetrics
from agent.agent_order import OrderManagementSystem as OMS
from feature_engineering.market_features import MarketFeatures


class AcquisitiontionTrader:
    """
    Test Backtest Engine.
    """

    def __init__(self, acc_volume: int = 1_000_000_0000):

        self.metrics = AgentMetrics()
        self.agent_trade = AgentTrade.history
        self.mi = MarketInterface()
        self.counter = 0
        self.acc_volume = acc_volume
        self.market_features = MarketFeatures()

    def apply_strategy(self):

        self.counter = self.counter + 1

        # -- submission

        if self.counter == 10:
            # get l2 in df form
            l2_df = self.market_features.level_2(store_timestamp=False,
                                                 data_structure='df')
            print(l2_df.T)

            # select the ask price of the 5th level as aggressive bid limit
            aggressive_limit = l2_df.loc[0,'l5-ask-price']

            self.mi.submit_order(side=1,
                                 limit=aggressive_limit,
                                 quantity=self.acc_volume)

if __name__ == '__main__':

    # inst: -> generate episode start list
    replay = Replay(identifier="FME",
                    start_date="2022-02-16",
                    end_date="2022-02-16",
                    episode_length="15m",
                    frequency="1m",
                    seed=42,
                    shuffle=True,
                    random_identifier=False,
                    exclude_high_activity_time=True,
                    )

    replay.base_reset()  # -> build new episode

    acc_trader = AcquisitiontionTrader()

    print("Episode Len: ", replay.episode.__len__())

    for i in range(210):  # replay_episode.episode.__len__()):

        replay.normal_step()
        acc_trader.apply_strategy()

        # observe what is happening...
        #if i % 10 == 0:
        #    # log market
        #print('step: ', i)
        #print("ORDER LIST", OMS.order_list)
        #print('Trade List', len(AgentTrade.history))
        print('Market:', Market.instances['ID'].state_l1)
