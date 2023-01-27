Level-3 Backtest Engine with RL extension.
------------------------------------------

Level-3 Backtest Engine based on Deutsche Börse A7 EOBI Data with an 
RL-extension in line with the OpenAI custom environment conventions which
is compatible with major RL frameworks.

Strategies can be defined in an agent class and passed as input argument to
the Replay class. Thereby, simulated orders can be submitted/cancelled/modified 
via the MarketInterface module. Replay is used to specify a backtest. During 
the backtest Replay builds new episodes with the Episode class which is the 
access point to the database. The matching engine simulation is implemented
in the Market module. Live market and agent data can be accessed via 
MarketMetrics and AgentMetrics. Time series data can be accessed via the
Context module. The FeatureEngineering module can be used to implement custom 
features. MarketTrade stores historical market executions. Reconstruction
translates messages into market updates. 

The RL-extension provides the abstract classes BaseObservationSpace, 
BaseReward, BaseActionSpace which can be subclassed for the development of 
custom RL-agents. The transition package can be used for the implementation
of the MDP. 