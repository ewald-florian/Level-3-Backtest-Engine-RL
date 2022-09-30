#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from reinforcement_learning.environment.environment import Environment
from reinforcement_learning.rl_agents.sample_agent import RlAgent
from replay_episode.replay import Replay
from utils.result_path_generator import generate_result_path

# generate pathname to store results
result_file = generate_result_path(name='otto')

# Start a new instance of Ray
ray.init()

agent = RlAgent(verbose=False)
# instantiate replay_episode and pass agent object as input argument
replay = Replay(rl_agent=agent,
                episode_length="1m")

# prepare config dict for the trainer set-up
config = PPO_DEFAULT_CONFIG
config["env"] = Environment
config["env_config"] = {
    "config": {
        "replay_episode": replay},
}

config["num_workers"] = 0
config["disable_env_checking"] = False
# TODO: 'horizon', 'soft_horizon' what is the difference?
#  'soft' horizon ist sehr weird, startet die ganze zeit neue episoden und
#  macht aber nichts...
#config['horizon'] = True
config['batch_mode'] = 'complete_episodes'

# Instantiate the Trainer object using above config.
rllib_trainer = PPOTrainer(config=config)
# print policy model
#print(rllib_trainer.get_policy().model.base_model.summary())

# result storage
results = []
episode_data = []
episode_json = []

# run training loops
num_iterations = 10
for n in range(num_iterations):

    # TODO: set-up trainer in a way that it resets after the done flag is true
    #  the episode was automatically limited to 2000 steps...
    result = rllib_trainer.train()

    results.append(result)
    # store relevant metrics from the result dict to the episode dict
    print('(TRAINER) Result')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)

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

"""
# Further Config Settings:
--------------------------
# --change model settings
config["model"]["fcnet_hiddens"] = [512, 512]
config["framework"] = "tf" 

# lstm
#config["model"]["use_lstm"] = True
#config["model"]["max_seq_len"] = 20
#config["model"]["lstm_cell_size"] = 256

# conv

config["num_workers"] = 2
config["evaluation_interval"] = 1
config["evaluation_duration"] = 1
config["evaluation_duration_unit"] = "episodes"
config["ignore_worker_failures"] = True

# test some parameters
#config["gamma"] = 1
#config["lr"] =  5e-05
config["train_batch_size"] = 250      # default = 4000
config["sgd_minibatch_size"] = 50

# custom model:
"""