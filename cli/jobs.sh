#!/bin/sh
cd
# go to directory
cd Level-3-Backtest-Engine-RL/reinforcement_learning/train_loops

# run first trainer
python -m rllib_trainer_template

# run second trainer
# ...

# Note:
# make executable: $chmod u+x jobs.sh
# can be executed in terminal with $./jobs.sh