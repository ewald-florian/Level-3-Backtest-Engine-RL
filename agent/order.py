#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 05/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Order Module for the Level-3 backtest engine"""
# ---------------------------------------------------------------------------

class Order:

    order_list = list()  # instance store

    # TODO: I could append all the order attributes in addition to the message
    def __init__(self, message):
        """
        Store Messages to order history list.
        """

        self.message = message
        # append order to history_list class attribute
        self.__class__.order_list.append(self)

    def _assert_params(self):
        pass

    def execute(self):
        # Change status, e.g. from 'ACTIVE' to 'FILLED'
        pass

    def cancel(self):
        # Change Status to 'CANCELLED'
        pass

    def __str__(self):
        pass

    @property
    def df_representation(self):
        pass

    @classmethod
    def reset_history(class_reference):
        """
        Reset order_list.
        """
        del class_reference.order_list[:]