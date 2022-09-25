#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ---------------------------------------------------------------------------
"""Template for Rllib training loops."""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-09-25"
__version__ = "0.1"
# ----------------------------------------------------------------------------
import json
import pprint

import pandas as pd

# rllib imports
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG

# library imports
from reinforcement_learning.environment import Environment
from reinforcement_learning.rl_agents.sample_agent import RlAgent
from replay.replay import Replay
from utils.result_path_generator import generate_result_path

# generate pathname to store results
result_file = generate_result_path(name='otto')

# Start a new instance of Ray
ray.init()

# instantiate agent
agent = RlAgent()
# instantiate replay and pass agent object as input argument
replay = Replay(rl_agent=agent)

# prepare config dict for the trainer set-up
config = PPO_DEFAULT_CONFIG
config["env"] = Environment
config["env_config"] = {
    "config": {
        "replay": replay},
}

config["num_workers"] = 0
config["disable_env_checking"] = False

# Instantiate the Trainer object using above config.
rllib_trainer = PPOTrainer(config=config)
# print policy model
#print(rllib_trainer.get_policy().model.base_model.summary())

# result storage
results = []
episode_data = []
episode_json = []

# run training loops
num_iterations = 1
for n in range(num_iterations):

    result = rllib_trainer.train()

    results.append(result)
    # store relevant metrics from the result dict to the episode dict
    episode = {
        "n": n,
        "episode_reward_min": result["episode_reward_min"],
        "episode_reward_mean": result["episode_reward_mean"],
        "episode_reward_max": result["episode_reward_max"],
        "episode_len_mean": result["episode_len_mean"],
    }

    episode_data.append(episode)
    episode_json.append(json.dumps(episode))

    # store results every iteration in case the training loop breaks
    result_df = pd.DataFrame(data=episode_data)
    result_df.to_csv(result_file, index=False)
    #file_name = rllib_trainer.save(checkpoint_root)

    print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_mean"]}')

result_df = pd.DataFrame(data=episode_data)
print(result_df.head())

result_df.to_csv(result_file, index=False)

# create checkpoint file to save trained weights
#checkpoint_file = rllib_trainer.save()
#print(100 * '-')
#print(f"Trainer (at iteration {rllib_trainer.iteration} was saved in '{checkpoint_file}'")

# shut down ray (important step since ray can occupy resources)
ray.shutdown()