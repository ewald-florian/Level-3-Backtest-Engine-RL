#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""Trainer Loop to test new abstract class structure of RL Agent

AGENTID = 200123
"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-12-05"
__version__ = "0.1"
# ----------------------------------------------------------------------------

# manage GPUs if executed on server
import platform
if platform.system() == 'Linux':
    gpuid = 'MIG-c8c5eee1-c148-5f66-9889-9759c8656d2b'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    from torch.cuda import device_count
    print('Number of Devices: ', device_count())

# build ins
import json
import pprint

# general imports
import pandas as pd

# rllib imports
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
# Note: DEFAULT_CONFIG is depreciated.
# from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG

# library imports
from reinforcement_learning.environment.tradingenvironment import \
    TradingEnvironment
from reinforcement_learning.agent_prototypes.twap_incentive_agent_reduced \
    import TwapIncentiveAgentReduced
from reinforcement_learning.agent_prototypes.twap_incentive_agent \
    import TwapIncentiveAgent
from replay_episode.replay import Replay
from utils.result_path_generator import generate_result_path
from utils.episode_stats_path_generator import generate_episode_stats_path
from reinforcement_learning.environment.episode_stats import EpisodeStats

# -- Create paths and files to store information.

# generate pathname to store results
result_file = generate_result_path(name='lstm_10_sec_wait')
print("RESULT_FILE:", result_file)
# generate json file to store episode statistics.
stats_path = generate_episode_stats_path(name='lstm_10_sec_wait')
EpisodeStats(stats_path)
print("EP_STATS_FILE:", EpisodeStats.path_name)

# -- Set up the training configuration.

# Start a new instance of Ray
ray.init()

# **SET THE EPISODE LENGTH FOR AGENT AND REPLAY**
episode_length = "10s"
agent = TwapIncentiveAgentReduced(verbose=True, episode_length=episode_length)
# instantiate replay_episode and pass agent object as input argument
replay = Replay(rl_agent=agent, episode_length=episode_length)

# prepare config dict for the trainer set-up
config = {}   # depreciated: config = PPO_DEFAULT_CONFIG
config["env"] = TradingEnvironment
config["env_config"] = {
    "config": {
        "replay_episode": replay},
}
# Size of the observation space
config["env_config"]["observation_size"] = 4  # Only agent observation.
config["env_config"]["action_size"] = 17  # Includes TWAP qt.

config["num_workers"] = 0
config["disable_env_checking"] = False

# lstm
config["model"] = {}
config["model"]["use_lstm"] = True
config["model"]["max_seq_len"] = 20
config["model"]["lstm_cell_size"] = 256

# set framework
config["framework"] = "tf2"


# Instantiate the Trainer object using above config.
rllib_trainer = PPOTrainer(config=config)
# print policy model
#print(rllib_trainer.get_policy().model.base_model.summary())

# result storage
results = []
episode_data = []
episode_json = []

# -- Run training loops.
num_iterations = 10

for n in range(num_iterations):

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
#######
print("Trainer:", rllib_trainer)
checkpoint_file = rllib_trainer.save()
print(f"Trainer (at iteration {rllib_trainer.iteration} was saved in '{checkpoint_file}'!")

# Here is what a checkpoint directory contains:
print("The checkpoint directory contains the following files:")
import os
os.listdir(os.path.dirname(checkpoint_file))
######

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

# config['horizon'] = True
# config['batch_mode'] = 'complete_episodes'

# custom model:
"""