#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 18/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Feature Engineering clas for the Level-3 backtest engine"""


# ---------------------------------------------------------------------------

# TODO: Enginering of features for the Optimal Execution Task
#TODO: Engineering of features for the cancellation task (spcific attributes
# of each active order, e.g. age, distance to original limit level / distance
# to original midprice, partial executions so far, relation to later placed
# orders etc.

class AgentFeatures:

    def __init__(self):
        pass

    # optimal execution
    def remaining_inventory(self):
        pass

    def elapsed_time(self):
        pass

    def remaining_time(self):
        pass

    def time_since_last_submission(self):
        pass

    def number_of_submissions(self):
        pass

    def inventory(self): # postion
        pass


    def reset(self):
        pass