#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Loop for Final Agent 1. Model 1: 128
AGENTID = 012823
"""

# build ins
import json
import pprint
import os
import platform

# general imports
import pandas as pd

# rllib imports
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune

# library imports
from reinforcement_learning.environment.tradingenvironment import \
    TradingEnvironment
from reinforcement_learning.agent_prototypes.final_agent_1 \
    import FinalOEAgent1

from replay_episode.replay import Replay
from utils.result_path_generator import generate_result_path
from utils.episode_stats_path_generator import generate_episode_stats_path
from utils.string_generator import generate_string
from reinforcement_learning.environment.episode_stats import EpisodeStats

# manage GPUs if executed on server.
if platform.system() == 'Linux':
    gpuid = 'MIG-ac3c47b3-456e-56ff-aa3e-5731e429d659'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

import tensorflow as tf

print("Num GPUs Available TF: ", len(tf.config.list_physical_devices('GPU')))

# SET UP TRAIN LOOP
# episode length for agent and replay

# Provide checkpoint path if trainer should be restored.
# TODO: ADD CHECKPOINT PATH
restoring_checkpoint_path = "/home/jovyan/ray_results/PPO_TradingEnvironment_2023-01-31_02-53-11byvd0295/checkpoint_000581"
name = 'final_agent_1_fcn_128_IS_REWARD_BAY_RUN1_'

num_iterations = 200
save_checkpoints_freq = 10
print_results_freq = 10
# environment.
episode_length = "10s"  # "60s", "30s"
obs_size = 40  # 40
action_size = 12

# fcnet.
fcnet_hiddens = [128, 128]
fcnet_activation = 'relu'
# lstm.
use_lstm = False
max_seq_len = None  # default 20
lstm_cell_size = None  # default 256
# training
learning_rate = 5e-05  # default 5e-05
# TODO
lr_schedule = [
    [0, 1.0e-6],
    [1, 1.0e-7]]
gamma = 1  # 0.99
train_batch = 2560  # 2560  # 4000  # default 4000
mini_batch = 128  # default: 128
rollout_fragment_length = 1280
num_workers = 2
num_envs_per_worker = 1
framework = "tf"
#  If batch_mode is “complete_episodes”, rollout_fragment_length is ignored.
batch_mode = 'complete_episodes'  # 'truncate_episodes'
# other settings.
disable_env_checking = False
print_entire_result = False  # contains a lot of useless info.
rllib_log_level = 'WARN'  # WARN, 'DEBUG'
# instantiate agent.
agent = FinalOEAgent1(verbose=False,
                      episode_length=episode_length,
                      initial_inventory_level="Avg-10s-Vol",
                      )

# Generate A string which contains all relevant infos for path names.
training_name = generate_string(
    name,
    episode_length,
    fcnet_hiddens,
    fcnet_activation,
    use_lstm,
    max_seq_len,
    lstm_cell_size,
    learning_rate,
    gamma,
    train_batch,
    mini_batch,
    batch_mode,
)

# -- Create paths and files to store information.

# generate pathname to store results
result_file = generate_result_path(name=training_name)
all_episode_result_file = generate_result_path(name=training_name + "_all_eps")
print("RESULT_FILE:", result_file)
print("ALL_EPS_RESULT_FILE:", all_episode_result_file)
# generate json file to store episode statistics.
stats_path = generate_episode_stats_path(name=training_name)
EpisodeStats(stats_path)
print("EP_STATS_FILE:", EpisodeStats.path_name)

# -- Set up the training configuration.

# Start a new instance of Ray (Note: num_gpus=1 required for server)
ray.init(num_gpus=1)

# instantiate replay_episode and pass agent object as input argument
replay = Replay(rl_agent=agent,
                episode_length=agent.episode_length,
                # Note: saved for later when I run on several symbols.
                # Testset:
                #identifier_list=['BAY', 'SAP', 'LIN', 'ALV', 'DTE'],
                #random_identifier=True,
                #start_date="2021-01-01",
                #end_date="2021-01-08",#"2021-04-30",
                #shuffle=True,
                #####
                verbose=False)

# -- Generate config file for PPO trainer.

# -- Environment Set-up.
config = {}
config["env"] = TradingEnvironment
config["env_config"] = {
    "config": {
        "replay_episode": replay},
}
# Size of the observation space
config["env_config"]["observation_size"] = obs_size
config["env_config"]["action_size"] = action_size

# -- PPO SETUP
# Notes: 0:  use the learner GPU for inference.
config["num_workers"] = num_workers
if num_workers > 0:
    config["num_envs_per_worker"] = num_envs_per_worker
# config["ignore_worker_failures"] = True
# config["num_envs_per_worker"] = 1
# Horizon: max time steps after which an episode will be terminated.
#  Note this limit should never be hit when everything works.
config["horizon"] = 100_000
# Note: these GPU settings are only required for several workers.
# config["num_gpus"] = 1
# config["num_cpus_per_worker"] = 1
config["disable_env_checking"] = disable_env_checking
config["log_level"] = rllib_log_level

# -- Model

config["framework"] = framework
# Note: tf2 and eager tracing do not work on server.
# config["eager_tracing"] = False
config["model"] = {}
# config["model"]["num_layers"] = num_layers
config["model"]["fcnet_hiddens"] = fcnet_hiddens
config["model"]["fcnet_activation"] = fcnet_activation
# discount factor.
config["gamma"] = gamma
# learning rate.
config["lr"] = learning_rate
# config["lr_schedule"] = lr_schedule
# Training batch size.
config["train_batch_size"] = train_batch  # default = 4000
# Mini batch size.
config["sgd_minibatch_size"] = mini_batch
# Batch mode
config['batch_mode'] = batch_mode
# rollout_fragment_length
config["rollout_fragment_length"] = rollout_fragment_length

# -- LSTM.
if use_lstm:
    config["model"]["use_lstm"] = use_lstm
    config["model"]["max_seq_len"] = max_seq_len
    config["model"]["lstm_cell_size"] = lstm_cell_size

# -- Instantiate the Trainer object using above config.
rllib_trainer = PPOTrainer(config=config)

# -- Reload from checkpoint.
if restoring_checkpoint_path:
    rllib_trainer.restore(restoring_checkpoint_path)
    print("RESTORED FROM CHECKPOINT, iterations: ", rllib_trainer._iteration)

# print policy model
# print(rllib_trainer.get_policy().model.base_model.summary())
# print total iteration to check if the trainer is new or reloaded.
# print(rllib_trainer._episodes_total)

# result storage list.
results = []
episode_data = []
episode_json = []
# Dict to store infos about each individual training episode.
all_episode_results = {"train_iteration": [],
                       "episode_len": [],
                       "episode_reward": []}

# -- Run training loops.

for iteration in range(num_iterations):

    # train and get results.
    result = rllib_trainer.train()
    # Store progress info for all episodes of the train iteration.
    all_ep_lengths = result['hist_stats']['episode_lengths']
    all_ep_rewards = result['hist_stats']['episode_reward']
    train_iter = [iteration] * len(all_ep_lengths)
    all_episode_results["episode_len"].extend(all_ep_lengths)
    all_episode_results["episode_reward"].extend(all_ep_rewards)
    all_episode_results["train_iteration"].extend(train_iter)
    # Store complete result to results list.
    results.append(result)
    # store relevant iteration metrics from the result dict to the episode dict
    results.append(result)
    episode = {
        "n": iteration,
        "episode_reward_min": result["episode_reward_min"],
        "episode_reward_mean": result["episode_reward_mean"],
        "episode_reward_max": result["episode_reward_max"],
        "episode_len_mean": result["episode_len_mean"],
    }

    episode_data.append(episode)
    episode_json.append(json.dumps(episode))

    # store results every iteration in case the training loop breaks.
    # - iteration level.
    result_df = pd.DataFrame(data=episode_data)
    result_df.to_csv(result_file, index=False)
    # - episode level.
    all_eps_df = pd.DataFrame(all_episode_results)
    result_df.to_csv(all_episode_result_file, index=False)

    # Save checkpoint every x iterations.
    if iteration % save_checkpoints_freq == 0:
        checkpoint_file = rllib_trainer.save()
        print("Checkpoint_File", checkpoint_file)

    if iteration % print_results_freq == 0:
        checkpoint_file = rllib_trainer.save()
        print(f'{iteration:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_mean"]}')

        # print results.
        if print_entire_result:
            print('(TRAINER) Result Iteration:', rllib_trainer.iteration)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(results)

# -- Store to df and Print Results.
result_df = pd.DataFrame(data=episode_data)
print(result_df.tail(5))
result_df.to_csv(result_file, index=False)
# -- Store each episode results to df and print
all_eps_df = pd.DataFrame(all_episode_results)
result_df.to_csv(all_episode_result_file, index=False)

# Save trainer to checkpoint file.
checkpoint_file = rllib_trainer.save()
print(f"Trainer (at iteration {rllib_trainer.iteration}) was "
      f"saved in '{checkpoint_file}'!")

# Shut down ray.
ray.shutdown()

"""
# NOTES

# Evaluation
config["evaluation_interval"] = 1
config["evaluation_duration"] = 1
config["evaluation_duration_unit"] = "episodes"

Agent Information:
 print(rllib_trainer._episodes_total)
 print(rllib_trainer._iteration)
 print(rllib_trainer.agent_timesteps_total)
 print(rllib_trainer._time_total)
"""
