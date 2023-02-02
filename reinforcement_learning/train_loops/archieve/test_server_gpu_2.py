#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
"""Trainer Loop to test new abstract class structure of RL Agent

AGENTID = 161122
"""
# ----------------------------------------------------------------------------
__author__ = "florian"
__date__ = "2022-11-17"
__version__ = "0.1"
# ----------------------------------------------------------------------------

# manage GPUs if executed on server
import platform
if platform.system() == 'Linux':
    gpuid = 'MIG-9985302b-0b2c-5903-929d-eb0313c73e0c'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    from torch.cuda import device_count
    print('Number of Devices: ', device_count())

# build ins
import json

# general imports

# rllib imports
import ray
from ray.rllib.agents.ppo import PPOTrainer
from utils.episode_stats_path_generator import generate_episode_stats_path
from reinforcement_learning.environment.episode_stats import EpisodeStats

# library imports
from reinforcement_learning.environment.tradingenvironment import TradingEnvironment
#from reinforcement_learning.agent_prototypes.sample_agent import RlAgent
from reinforcement_learning.agent_prototypes.archieve.twap_incentive_agent \
    import TwapIncentiveAgent
from replay_episode.replay import Replay
from utils.result_path_generator import generate_result_path

import tensorflow as tf
print("Num GPUs Available TF: ", len(tf.config.list_physical_devices('GPU')))


# generate pathname to store results
# NOTE: contains sys-if condition for base_dir...
result_file = generate_result_path(name='gputest')
stats_path = generate_episode_stats_path(name='gputest')
EpisodeStats(stats_path)
print("EP_STATS_FILE:", EpisodeStats.path_name)

print("RESULT_FILE", result_file)

# Start a new instance of Ray
# TODO: Try to set num_gpus in ray.init!
ray.init(num_gpus=1)

# agent = MoreActionsAgent(verbose=False)
agent = TwapIncentiveAgent(verbose=False)
# instantiate replay_episode and pass agent object as input argument
replay = Replay(rl_agent=agent,
                episode_length="1m",
                verbose=True)

# prepare config dict for the trainer set-up
config = {}
config["env"] = TradingEnvironment
config["env_config"] = {
    "config": {
        "replay_episode": replay},
}
# Size of the observation space
config["env_config"]["observation_size"] = 34
config["env_config"]["action_size"] = 17

config["num_workers"] = 0
config["disable_env_checking"] = False

config['batch_mode'] = 'complete_episodes'

# TEST TF2 Framework GPU support.
config["framework"] = "tf2"
# TODO: test eager tracing
config["eager_tracing"] = False
# NOTE: torch hat auch keine GPU genutzt
#config["framework"] = "torch"
rllib_log_level = 'DEBUG'  # WARN, 'DEBUG'

rllib_trainer = PPOTrainer(config=config)

# result storage
results = []
episode_data = []
episode_json = []

# run training loops
num_iterations = 2
for n in range(num_iterations):

    print("iteration: ", n)
    result = rllib_trainer.train()

    results.append(result)

    episode = {
        "n": n,
        "episode_reward_min": result["episode_reward_min"],
        "episode_reward_mean": result["episode_reward_mean"],
        "episode_reward_max": result["episode_reward_max"],
        "episode_len_mean": result["episode_len_mean"],
    }

    episode_data.append(episode)
    episode_json.append(json.dumps(episode))

    print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_mean"]}')


ray.shutdown()