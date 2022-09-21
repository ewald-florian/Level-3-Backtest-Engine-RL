#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 21/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Prototype for RL Agents.
"""
# ---------------------------------------------------------------------------

"""
There are two possible set-ups for a replay-step

1. Everything happens inside the replay.step() method:

def replay.step(self)

    if self.step > (len(self.episode)):
        self.done = True

    self.take_action(action)
    
    observation = ObservationSpace.observation
    
    reward = Reward.reward() 
    
    info = {'pnl': AgentMetrics.pnl, 'num_trades':AgentMetrics.num_trades}
    
    return observation, reward, done, info
"""