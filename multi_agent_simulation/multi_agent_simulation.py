"""

MultiAgentReplay class

Three alternative Set-ups / Modes:

1.	Hybrid: run a normal historical backtest and let several agents submit
impact orders ("replay the flash-crash but with..."-scenarios, or just to have
enough market activity for anything interesting to happen)

2.	Pure simulation from snapshot: The initial state is build from a snapshot
but the following trading activity comes entirely from the agents (no message
data)

3.	Start with a plain orderbook and let agents submit and cancel orders.

-> Each agent needs his own instances of OMS and AgentTrade
-> Or, the class attributes are dicts and the agent can store their personal
information under their name as key:
[AgentTrade.history = {"agent1": [trade_1,...], "agent2": [trade_1,...], ...}

-> Or the information is marked with additional key:
trade = {"price":10, "quantity" : 22, "side"=1, "agent_id"=agent_1}

The "WakeUpMachine" wakes agents according to a predefined stochastic process,
includes stochastic timing of market updates (the market cannot just be updated
every nanosecond or so but should tick similar to a real market with some
variation and a realistic average interval between events.

=> The WakeUpMachine can become a RL agent (or some kind of machine learning
algorithm) which learns to generate realistic market activity by waking up
agents. Maybe it can learn to model different market regimes (e.g. high
activity time vs. normal time, crash scenarios etc.)

Scalable NoiseTrades, their metrics are less relevant they should mostly be
able to generate a lot of "realistic" market activity
- > Could be many noise trader instances
- > or one instance that trades a lot and represents many market participants

MultiAgentMetrics
- > collect metrics of all agents in one datastructue for good overview of
results

The Simulation can use many of the normal backtest classes so it will just be
an additional feature that can optionally be used!

I could peace by peace add additional features to the module:
- heterogeneous latency (this one is easy)
- circuit breakers
- manual impact events

Theoretically, the simulation can be even used to analyze the outcomes of
changes in the market design (ticksize, priority-rules, order types). However,
results will always be strongly dependent on initial set-up of the model and
hence difficult to generalize.

------------
import Market
import Replay

import Agent_1
import Agent_2
...
import Agent_n

# run simulation

market = Market()
replay = Replay()

agent_n = Agent_n()

replay.reset()

for step in steps:

    - update market (if hybrid)
    - wake up agent X (according to a WakeUpMachine)
    - agent X can submit/cancel with impact according to his strategy
    - match orderbook


"""