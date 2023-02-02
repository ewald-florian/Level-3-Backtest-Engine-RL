#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST LOOP for Final Agent 2. Model 1: 128
AGENTID = 012823

10 Observations, 6 Actions
"""

# build ins
import json
import pprint
import os
import platform
import copy

# general imports
import pandas as pd
import numpy as np

# rllib imports
import ray
from ray.rllib.agents.ppo import PPOTrainer
import tensorflow as tf

# library imports
from reinforcement_learning.environment.tradingenvironment import \
    TradingEnvironment
from reinforcement_learning.agent_prototypes.final_agent_2_limited \
    import FinalOEAgent2Limited
from reinforcement_learning.agent_prototypes.final_agent_1 import \
    FinalOEAgent1

from replay_episode.replay import Replay
from utils.test_result_path import generate_test_result_path

# -- The TEST LOOP is initialized exactly as the TRAIN Loop since the correct
# -- config file specifications are required to restore the weights.

# manage GPUs if executed on server.
"""
if platform.system() == 'Linux':
    gpuid = 'MIG-b33f9985-2600-590d-9cb1-002ae4ce5957'
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
"""

print("Num GPUs Available TF: ", len(tf.config.list_physical_devices('GPU')))

# -- Set Up.
# "A1_FCN_128"
# "A1_FCN_128_LSTM"
# "A1_FCN_256"
# "A1_FCN_256_LSTM"
# "A1_BAY"
# "A2_LIMITED"
#
# ----------------------------
# TODO: Insert Agent Name
STRATEGY_NAME = "A1_FCN_128"
AGENT = FinalOEAgent1  # FinalOEAgent2Limited # FinalOEAgent1
SYMBOL = "DTE"
# ----------------------------
TEST_START = "2021-05-14"
TEST_END = "2021-06-30"
NUM_ITERS_STORE_RESULTS = 250
VERBOSE = True
NUM_TEST_EPISODES = 1_000_000
# ----------------------------
print(80*"*")
print("TEST LOOP STARTED")
print("STRATEGY: ", STRATEGY_NAME)
print("SYMBOL: ", SYMBOL)
print("Start:", TEST_START)
print("End:", TEST_END)
print(80*"*")

# Paths to base config dicts.
if platform.system() == 'Darwin':  # macos
    base_config_path = "/Users/florianewald/PycharmProjects/Level-3-Backtest-"\
                       "Engine-RL/reinforcement_learning/base_configs/"

elif platform.system() == 'Linux':
    base_config_path = "/home/jovyan/Level-3-Backtest-" \
                       "Engine-RL/reinforcement_learning/base_configs/"

base_config_path = base_config_path + STRATEGY_NAME + "_base_config.json"
with open(base_config_path, 'r') as fp:
    base_config = json.load(fp)

# Match base config and checkpoint path depending on strategy name.
if STRATEGY_NAME == "A1_FCN_128":
    CHECKPOINT_PATH = "/home/jovyan/ray_results/PPO_TradingEnvironment_2023-02-01_17-27-0227wpvlpg/checkpoint_001374"

elif STRATEGY_NAME == "A1_FCN_128_LSTM":
    CHECKPOINT_PATH = "/home/jovyan/ray_results/PPO_TradingEnvironment_2023-02-01_18-07-53kd5zyj95/checkpoint_001314"

elif STRATEGY_NAME == "A1_FCN_256":
    # CHECKPOINT_PATH = "/Users/florianewald/Downloads/A1_FCN_256_final_checkpoint"
    CHECKPOINT_PATH = "/home/jovyan/ray_results/PPO_TradingEnvironment_2023-02-01_16-29-5770ets8gb/checkpoint_001102"  # works!

elif STRATEGY_NAME == "A1_FCN_256_LSTM":
    CHECKPOINT_PATH = "/home/jovyan/ray_results/PPO_TradingEnvironment_2023-02-01_16-21-24t0r6c3vt/checkpoint_001012"

elif STRATEGY_NAME == "A1_BAY":
    CHECKPOINT_PATH = ...

elif STRATEGY_NAME == "A2_LIMITED":
    CHECKPOINT_PATH = "/Users/florianewald/ray_results/PPOTrainer_TradingEnvironment_2023-02-02_12-40-48qhxzjq6z/checkpoint_000925/checkpoint-925"
    #CHECKPOINT_PATH = "/Users/florianewald/ray_results/PPOTrainer_TradingEnvironment_2023-01-31_22-31-19agy3bqbz/checkpoint_000401/checkpoint-401"

# Start ray.
ray.init(num_gpus=0, num_cpus=1)
# ray.init(num_gpus=1)

# Initialize agent for Replay.
agent = AGENT(verbose=False,
              episode_length="10s",
              initial_inventory_level="Avg-10s-Vol",
              )
# Initialize Replay for the TEST-RUN.
replay = Replay(rl_agent=agent,
                episode_length="10s",
                frequency="5m",
                identifier=SYMBOL,
                start_date=TEST_START,
                end_date=TEST_END,
                shuffle=False,
                verbose=False)

# Extend base config with instances.
base_config["env"] = TradingEnvironment
base_config['env_config']['config']['replay_episode'] = replay
base_config["disable_env_checking"] = True
#base_config["framework"] = "tf"

print("(INSTANTIATED) FROM {}".format(base_config_path))
trained_strategy = PPOTrainer(config=base_config)
trained_strategy.restore(CHECKPOINT_PATH)
print("(RESTORED) RESTORED AGENT WITH {} ITERATIONS".format(
    trained_strategy.iteration))
print("(RESTORED) FROM CHECKPOINT: {}".format(CHECKPOINT_PATH))

# -- Result path.

results = []
result_path = generate_test_result_path(symbol=replay.identifier,
                                        strategy_name=STRATEGY_NAME)
print("(INFO) TEST RESULTS WILL BE STORED TO: ", result_path)

# -- Test loop.

# Instantiate environment.
env = TradingEnvironment(base_config["env_config"])
# Reset env, get initial obs.
obs = env.reset()

# Dict to store rewards for each test-episode.
reward_dict = {}
episode_counter = 0
episode_reward = 0


# Try excet since eventually the episode start list will be other.
try:

    while episode_counter < NUM_TEST_EPISODES:
        # Compute action.
        action = trained_strategy.compute_single_action(
            observation=obs,
            explore=False,
            # TODO: warum habe ich das nochmal gemacht?
            policy_id="default_policy"
        )
        # Send action to env and take step receiving obs, reward, done, info.
        obs, reward, done, info = env.step(action)
        # Count the reward.
        episode_reward += reward

        # If episode is done, collect stats and reset env.
        if done:
            # -- Store results.

            # Get results from env.
            reward = episode_reward
            episode_start = copy.deepcopy(env.replay.episode.episode_start)
            overall_is = copy.deepcopy(
                env.replay.rl_agent.agent_metrics.overall_is(scaling_factor=1))
            vwap_sell = copy.deepcopy(
                env.replay.rl_agent.agent_metrics.vwap_sell)
            total_episode_steps = copy.deepcopy(env.replay.episode._step)

            # Append results to results list.
            results.append(np.array([episode_start,
                                     overall_is,
                                     vwap_sell,
                                     total_episode_steps,
                                     reward]))

            if episode_counter % NUM_ITERS_STORE_RESULTS == 0:
                df = pd.DataFrame(results, columns=["episode_start",
                                                    "overall_is",
                                                    "vwap_sell",
                                                    "total_steps",
                                                    "reward"])
                df.to_csv(result_path, index=False)

                # Print to terminal.
                print(df)

            # -- Reset the environment to run the next episode.
            obs = env.reset()
            episode_counter += 1
            # Reset the reward.
            episode_reward = 0

            # DEBUGGING
            # print("initial inventory", env.replay.rl_agent.initial_inventory)
            # print("episode start: ", env.replay.episode.episode_start)
            # print("identifier: ", env.replay.episode.identifier)

# Store results when the loop fails since episodes are over.
except:
    # -- Store Final Results.
    # Store final  results to DF:
    df = pd.DataFrame(results, columns=["episode_start",
                                        "overall_is",
                                        "vwap_sell",
                                        "total_steps",
                                        "reward"])
    df.to_csv(result_path, index=False)


# Redundantly store results again to be safe.

# -- Store Final Results.
# Store final  results to DF:
df = pd.DataFrame(results, columns=["episode_start",
                                    "overall_is",
                                    "vwap_sell",
                                    "total_steps",
                                    "reward"])
df.to_csv(result_path, index=False)

print("(INFO) TEST RUN COMPLETE")
print("(INFO) RESULTS STORED IN: ", result_path)

# Shut down ray.
ray.shutdown()
