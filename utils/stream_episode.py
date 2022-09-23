#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Class to stream episodes, not directly related to backtesting but useful for
data analysis, development, testing and debugging.

some use cases:
---------------
- replay specific episodes by streaming their messages
- print individual messages
- print various feature representations
- test new methods
- render the market activity in plots+
- search bugs and control if computations are correct
- get more intuitive understanding of message data
- get more intuitive understanding of the Level-3 Backtest Engine
- run backtest with very high logging level (print out every step)

"""
# ---------------------------------------------------------------------------


import time

import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)

from market.market_interface import MarketInterface
from market.market import Market
from replay.replay import Replay
from feature_engineering.market_features import MarketFeatures
from agent.agent_metrics import AgentMetrics
from agent.agent_order import OrderManagementSystem as OMS
from utils.plot_lob import plot_lob


class StreamEpisode:

    def __init__(self,
                 number_of_steps=2000,
                 sleep_time=1, # sec.
                 ):

        # static
        self.sleep_time = sleep_time
        self.number_of_steps = number_of_steps

        # dynamic
        self.step = 0

        self.replay = Replay()  # -> build new episode
        self.market = Market(market_id="ID", l2_levels=5, l3_levels=5)
        self.market_features = MarketFeatures()
        self.market_interface = MarketInterface()
        self.agent_metrics = AgentMetrics()

        self.replay.reset() # -> build new episode
        print("Number of Message Packets: ", self.replay.episode.__len__())

        self.template_id_map = {
            13100: "order_add",
            13101: "order_modify",
            13102: "order_delete",
            13103: "order_mass_delete",
            13104: "execution_full",
            13105: "execution_partial",
            13106: "order_modify_same_priority",
            13202: "execution_summary"
        }

    def stream(self):

        horizon = 2000

        for i in range(horizon):  # replay.episode.__len__()):

            self.replay.normal_step()
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

            self._plot_lob()

            self._print_messages()

            if self.step % 10 == 5:
                self._print_lob_dataframe()

            if self.step==5:
                self._submit_order(side=1)

            if self.step == 6:
                self._fast_run_session()

            if self.step==10:
                self._submit_order(side=2)

            if self.step % 10 == 0:
                print(self.agent_metrics)


            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            self.step += 1

            if self.sleep_time:
                time.sleep(self.sleep_time/2)

    def _fast_run_session(self, number_of_steps=1000):
        for step in range(number_of_steps):
            self.replay.normal_step()
            self.step +=1
            print('.')#, end="")

    def _print_messages(self):

        message_packet = self.replay.episode.message_packet_list[
            self.replay.episode._step]

        for message in message_packet:

            if message['MessageHeader']['TemplateID'] in [13005, 13004]:
                pass
            else:
                msg_type = self.template_id_map.get(
                    message['MessageHeader']['TemplateID'])
                print(80 * '-')
                print("message type: ", msg_type)
                if 'OrderDetails' in message.keys():
                    side = message['OrderDetails']['Side']
                    if side == 1:
                        side = "Buy"
                    else:
                        side = "Sell"
                    price = int(message['OrderDetails']['Price'])*1e-8
                    qt = int(message['OrderDetails']['DisplayQty'])*1e-4
                    timestamp = self._parse_unix_time(message['OrderDetails']
                                                 ['TrdRegTSTimePriority'])

                    print("time: {} | side: {} | price: {} | quantity: {} ".format(
                        timestamp, side, price, qt))

                print(45*'_')
                pp.pprint(message)
                print(80 * '-')
                time.sleep(3)

    @staticmethod
    def _parse_unix_time(timestamp, timespec = 'seconds'): # ‘nanoseconds’.

        unix = int(timestamp)
        datetime_timestamp = pd.to_datetime(unix, unit='ns')
        datetime_timestamp = datetime_timestamp.isoformat(timespec=timespec)
        return datetime_timestamp

    def _print_level_1(self):

        state_l1 = self.market.instances['ID'].state_l1
        print(state_l1)

    def _print_lob_dataframe(self):
        df = self.market_features.level_2(data_structure='df',
                                          store_timestamp=False)
        #parse prices and quantities

        df.iloc[:, ::2] = round(df.iloc[:,::2]*1e-8, 2)
        df.iloc[:, 1::2] = round(df.iloc[:, 1::2] * 1e-4, 2)
        df.index = ["Limit Order Book"]

        #df_new = pd.DataFrame(columns=["Level", "Price", "Quantity"])
        #df_new.loc["Level"] =
        #df_new.loc["Price"] = df.iloc[:, ::2].values()
        #df_new.loc["Quantity"] = df.iloc[:, 1::2]
        #print(df_new)
        print(df.T)

    # TODO: update plot every step instead of creating a new plot
    def _plot_lob(self):
        # imported function
        level2_dict = self.market.instances['ID'].state_l2
        plot1 = plot_lob(level2_dict)
        #plt.show()

    def _render_lob(self):
        # render lob in second window
        pass

    def _plot_midprice_series(self):
        # standard line plot to show price chart
        # or best-bid an best_ask
        pass

    def _submit_order(self, side=1):
        if side == 1:
            price = self.market_features.best_ask()
        else:
            price = self.market_features.best_bid()

        quantity = 77_0000

        self.market_interface.submit_order(side=1,
                                           limit=price,
                                           quantity=quantity)

        print(100*'-')
        agent_message = OMS.order_list[-1]
        print("Agent Order:")
        print(20*'_')
        pp.pprint(agent_message)
        print(100 * '-')




if __name__ == '__main__':

    streamer = StreamEpisode()

    streamer.stream()

