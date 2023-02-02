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

from replay_episode.replay import Replay
from utils.test_result_path import generate_test_result_path

# -- The TEST LOOP is initialized exactly as the TRAIN Loop since the correct
# -- config file specifications are required to restore the weights.

# manage GPUs if executed on server.
if platform.system() == 'Linux':
    gpuid = 'MIG-ac3c47b3-456e-56ff-aa3e-5731e429d659'
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

print("Num GPUs Available TF: ", len(tf.config.list_physical_devices('GPU')))

# -- Set Up.
# ---------------
CHECKPOINT_PATH = "/Users/florianewald/ray_results/PPOTrainer_TradingEnvironment_2023-01-31_22-31-19agy3bqbz/checkpoint_000401/checkpoint-401"
BASE_CONFIG = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/reinforcement_learning/base_configs/agent2_fcn128_base_config.json"
STRATEGY_NAME = "tttest"  # TODO: Make name with timestamp.now, symbol and strategy
SYMBOL = "BAY"
TEST_START = "2021-01-04"
TEST_END = "2021-01-08"
FREQUENCY = "5m"
NUM_ITERS_STORE_RESULTS = 100
VERBOSE = True
NUM_TEST_EPISODES = 100
PRINT_FREQUENCY = 10
# ----------------

# TODO: einfach über name laden
# Match base config and checkpoint path depending on strategy name.
if STRATEGY_NAME == "A1_FCN_128":
    BASE_CONFIG = ...
    CHECKPOINT_PATH = ...
elif STRATEGY_NAME == "A1_FCN_128_LSTM":
    BASE_CONFIG = ...
    CHECKPOINT_PATH = ...
elif STRATEGY_NAME == "A1_FCN_256":
    BASE_CONFIG = ...
    CHECKPOINT_PATH = ...
elif STRATEGY_NAME == "A1_FCN_256_LSTM":
    BASE_CONFIG = ...
    CHECKPOINT_PATH = ...
elif STRATEGY_NAME == "A1_BAY":
    BASE_CONFIG = ...
    CHECKPOINT_PATH = ...
elif STRATEGY_NAME == "A2_LIMITED":
    BASE_CONFIG = ...
    CHECKPOINT_PATH = ...

# -- Config.:
with open(BASE_CONFIG, 'r') as fp:
    base_config = json.load(fp)

# Start ray.
ray.init(num_gpus=1, num_cpus=1)

# Initialize agent for Replay.
agent = FinalOEAgent2Limited(verbose=False,
                             episode_length="10s",
                             initial_inventory_level="Avg-10s-Vol",
                             )
# Initialize Replay for the TEST-RUN.
replay = Replay(rl_agent=agent,
                episode_length="10s",
                identifier=SYMBOL,
                # Note: saved for later when I run on several symbols.
                # Testset:
                # identifier_list=['BAY', 'SAP', 'LIN', 'ALV', 'DTE'],
                # random_identifier=True,
                # start_date="2021-01-01",
                # end_date="2021-01-08",#"2021-04-30",
                # shuffle=True,
                #####
                verbose=False)

# Extend base config with instances.
base_config["env"] = TradingEnvironment
base_config['env_config']['config']['replay_episode'] = replay

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
        print(f"Episode done: Total reward = {episode_reward}")

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
