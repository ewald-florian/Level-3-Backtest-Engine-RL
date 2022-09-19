#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Order class for the Level-3 backtest engine"""


# ---------------------------------------------------------------------------

# Should later replace Market.agent_message_list

class OrderManagementSystem:  # OMS

    # Basically, Market.agent_message_list
    order_list = list()  # instance store

    def __init__(self, message, verbose=True):
        """
        Store agent messages (submissions and cancellations)
         in class attribute order_list.
        :param message
            dict, agent message
        :param verbose
            bool, set True to log messages.
        """

        # append order to order_list class attribute
        self.__class__.order_list.append(message)

        # TODO: besser hier loggen order direkt im MarketInterface?

        # logging
        if message['template_id'] == 99999:
            side = message["side"]
            limit = message["price"]
            quantity = message["quantity"]
            # TODO: include latency
            timestamp = message["timestamp"]
            # TODO
            # if verbose:
            #    print()
            # print(f'(INFO) Agent Order Submitted: Side: {side} | Price: {price} | Quantity: {quantity}')

        elif message['template_id'] == 66666:

            order_message_id = message['order_message_id']
            # TODO
            # if verbose:
            #    print(order_message_id)

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
    def reset_history(class_reference):
        """
        Reset order_list.
        """
        del class_reference.order_list[:]