#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to train and restore agent.
Technisch gesehen funktioniert es!
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
from reinforcement_learning.agent_prototypes.twap_incentive_agent_reduced \
    import TwapIncentiveAgentReduced
from reinforcement_learning.agent_prototypes.is_agent_2 import ISAgent2
from replay_episode.replay import Replay
from utils.result_path_generator import generate_result_path
from utils.episode_stats_path_generator import generate_episode_stats_path
from utils.string_generator import generate_string
from reinforcement_learning.environment.episode_stats import EpisodeStats

# manage GPUs if executed on server
if platform.system() == 'Linux':
    gpuid = 'MIG-c8c5eee1-c148-5f66-9889-9759c8656d2b'
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

import tensorflow as tf

print("Num GPUs Available TF: ", len(tf.config.list_physical_devices('GPU')))

# -- Part 1: Set up and train.

name = 'test_train_restore'
num_iterations = 2
save_checkpoints_freq = 10
print_results_freq = 10
# environment.
episode_length = "10s"  # "60s", "30s"
obs_size = 40  # 40
action_size = 12

# fcnet.
fcnet_hiddens = [128, 128]
fcnet_activation = 'tanh'
# lstm.
use_lstm = False
max_seq_len = None  # default 20
lstm_cell_size = None  # default 256
# training
# TODO: Teste default lr, erstelle lr schedule
learning_rate = 5e-05  # default 5e-05
# TODO
lr_schedule = [
    [0, 1.0e-6],
    [1, 1.0e-7]]
gamma = 1  # 0.99
# TODO: Teste größere batches! default ist 4000
train_batch = 32 # 4000  # default 4000
mini_batch = 8  # default: 128
rollout_fragment_length = 32
num_workers = 0
#  If batch_mode is “complete_episodes”, rollout_fragment_length is ignored.
batch_mode = 'complete_episodes'  # 'truncate_episodes'
# other settings.
disable_env_checking = False
print_entire_result = True
rllib_log_level = 'WARN'  # WARN, 'DEBUG'
# agent
agent = ISAgent2(verbose=True,
                 episode_length=episode_length,
                 initial_inventory=800_0000,
                 )

# Generate A string which contains all relevant infos.
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
print("RESULT_FILE:", result_file)
# generate json file to store episode statistics.
stats_path = generate_episode_stats_path(name=training_name)
EpisodeStats(stats_path)
print("EP_STATS_FILE:", EpisodeStats.path_name)

# -- Set up the training configuration.

# Start a new instance of Ray
ray.init(num_gpus=1)

# instantiate replay_episode and pass agent object as input argument
replay = Replay(rl_agent=agent,
                episode_length=agent.episode_length,
                verbose=False)

# -- Generate config file for PPO trainer.
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
# TODO: For efficient use of GPU time, use a small number of GPU workers and a
#  large number of envs per worker.
config["num_workers"] = num_workers
# config["ignore_worker_failures"] = True
# TODO: verstehen was genau das heißt, wird das env immer wieder resettet?
# config["num_envs_per_worker"] = 1
# Horizon: max time steps after which an episode will be terminated.
#  Note this limit should never be hit when everything works.
config["horizon"] = 100_000
# Make sure to use GPU.
# TODO: test on server!
# config["num_gpus"] = 1
# config["num_cpus_per_worker"] = 1
config["disable_env_checking"] = disable_env_checking
config["log_level"] = rllib_log_level
# set framework
config["framework"] = "tf"
#config["eager_tracing"] = False
# FCN size.
config["model"] = {}
config["model"]["fcnet_hiddens"] = fcnet_hiddens
config["model"]["fcnet_activation"] = fcnet_activation
# discount factor.
config["gamma"] = gamma
# learning rate.
# TODO: Include lr scheduler (decreasing lr over time)
config["lr"] = learning_rate
# config["lr_schedule"] = lr_schedule
# Training batch size.
# TODO: wieder einkommentieren
config["train_batch_size"] = train_batch  # default = 4000
# Mini batch size.
config["sgd_minibatch_size"] = mini_batch
# Batch mode
config['batch_mode'] = batch_mode
# rollout_fragment_length
#config["rollout_fragment_length"] = rollout_fragment_length

# -- Wrap LSTM around model if use_lstm is set True.
if use_lstm:
    config["model"]["use_lstm"] = use_lstm
    config["model"]["max_seq_len"] = max_seq_len
    config["model"]["lstm_cell_size"] = lstm_cell_size

# -- Instantiate the Trainer object using above config.
rllib_trainer = PPOTrainer(config=config)
# print policy model
# print(rllib_trainer.get_policy().model.base_model.summary())
# print total iteration to check if the trainer is new or reloaded.
# print(rllib_trainer._episodes_total)

# result storage list.
results = []
episode_data = []
episode_json = []

# -- Run training loops.

for iteration in range(num_iterations):

    # train and get results.
    result = rllib_trainer.train()

    # Store complete result to results list.
    results.append(result)
    # store relevant metrics from the result dict to the episode dict
    results.append(result)
    episode = {
        "n": iteration,
        "episode_reward_min": result["episode_reward_min"],
        "episode_reward_mean": result["episode_reward_mean"],
        "episode_reward_max": result["episode_reward_max"],
        "episode_len_mean": result["episode_len_mean"],
    }

    episode_data.append(episode)
    # TODO: Warum habe ich das nochmal gemacht?
    episode_json.append(json.dumps(episode))

    # store results every iteration in case the training loop breaks.
    result_df = pd.DataFrame(data=episode_data)
    result_df.to_csv(result_file, index=False)
    # Save checkpoint every x iterations.
    if iteration % save_checkpoints_freq == 0:
        checkpoint_file = rllib_trainer.save()

    if iteration % print_results_freq == 0:
        checkpoint_file = rllib_trainer.save()
        print(
            f'{iteration:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_mean"]}')

        # print results.
        if print_entire_result:
            print('(TRAINER) Result Iteration:', rllib_trainer.iteration)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(results)

# -- Store to df and Print Results.
result_df = pd.DataFrame(data=episode_data)
print(result_df.head(5))
result_df.to_csv(result_file, index=False)

# Save trainer to checkpoint file.
# TODO: name checkpoint file (or folder)
checkpoint_file = rllib_trainer.save()
print(
    f"Trainer (at iteration {rllib_trainer.iteration} was saved in '{checkpoint_file}'!")

# Here is what a checkpoint directory contains:
print("The checkpoint directory contains the following files:")
print(os.listdir(os.path.dirname(checkpoint_file)))

# Shut down ray.
ray.shutdown()

# -- Part 2: restore agent from file.
# --------------------------------
config_copy = config.copy()
config_copy["lambda"] = 0.666
del rllib_trainer
print("deleted old trainer")

# Create new rllib trainer.
# Note: this needs a correct config file!
new_trainer = PPOTrainer(config=config_copy)
print("created new trainer")
# Restore old agent:
new_trainer.restore(checkpoint_file)
print("restored trainer from:", checkpoint_file)

print(new_trainer.config)

# -- Part 3) Run the environment with the loaded agent in "test-mode"
# Instantiate environment.

# Manually set up trading environment.
env = TradingEnvironment(config["env_config"])
# Reset env, get initial obs.
obs = env.reset()

# Reset counters.
num_episodes = 0
episode_reward = 0.0
# Dict to store rewards for each test-episode.
reward_dict = {}

# How many episodes should be tested.
num_test_episodes = 1

while num_episodes < num_test_episodes:
    # Compute action from my trained policy.
    action = new_trainer.compute_single_action(
        observation=obs,
        explore=False,
        policy_id="default_policy"
    )
    # Send action to env.
    obs, reward, done, info = env.step(action)
    # Count the reward.
    episode_reward += reward

    # If episode is done, reset env and start new episode.
    if done:
        print(f"Episode done: Total reward = {episode_reward}")
        # Add to reward_dict.
        reward_dict[num_episodes] = episode_reward
        obs = env.reset()
        num_episodes += 1
        episode_reward = 0.0
        # TODO: store episode stats if done!
        #  ... use the infra structure EpisodeStats.


# Output reward_dict:
print(reward_dict)
