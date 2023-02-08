#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
""" Agent Order Management class for the Level-3 backtest engine"""

import pandas as pd


class OrderManagementSystem:
    """
    Management of agent orders during backtest. Order status is defined
    under dict key "template_id"
    """

    # Basically, Market.agent_message_list
    order_list = list()  # instance store

    def __init__(self,
                 message: dict,
                 verbose: bool = True):
        """
        Store agent messages (submissions and cancellations)
         in class attribute order_list.
        :param message
            dict, agent message
        :param verbose
            bool, set True to log messages.
        """

        self.verbose = verbose
        # append order to order_list class attribute
        self.__class__.order_list.append(message)

        # logging
        if message['template_id'] == 99999:
            side = message["side"]
            limit = message["price"]
            quantity = message["quantity"]
            timestamp = message["timestamp"]


        elif message['template_id'] == 66666:

            order_message_id = message['order_message_id']

    @property
    def dataframe(self):
        """
        Dataframe representation of trade history.
        """
        return pd.DataFrame.from_records(
            self.__class__.order_list)

    @property
    def array(self):
        """
        Numpy Array representation of trade history.
        """
        df = self.dataframe
        return np.array(df)

    @property
    def message_count(self):
        """
        Trade count
        """
        return len(self.__class__.order_list)

    @classmethod
    def reset_history(cls):
        """
        Reset order_list.
        """
        del cls.order_list[:]
