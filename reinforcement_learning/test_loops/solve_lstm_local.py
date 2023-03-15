#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Loop for Final Agent 1 with MODEL 2 (128-LSTM)
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

from replay.replay import Replay
from utils.result_path_generator import generate_result_path
from utils.episode_stats_path_generator import generate_episode_stats_path
from utils.string_generator import generate_string
from reinforcement_learning.environment.episode_stats import EpisodeStats

# manage GPUs if executed on server.
if platform.system() == 'Linux':
    gpuid = 'MIG-9985302b-0b2c-5903-929d-eb0313c73e0c'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

import tensorflow as tf

print("Num GPUs Available TF: ", len(tf.config.list_physical_devices('GPU')))

# SET UP TRAIN LOOP
# episode length for agent and replay

# Provide checkpoint path if trainer should be restored.
# TODO: Add Checkpoint Path
restoring_checkpoint_path = None #"/home/jovyan/ray_results/PPO_TradingEnvironment_2023-01-31_20-57-12hix01vid/checkpoint_000813/"
name = 'GET_CONFIG'
num_iterations = 1000
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

use_lstm = True
max_seq_len = 10  # default 20
lstm_cell_size = 10  # 128  # default 256
# training
learning_rate = 5e-05  # default 5e-05
gamma = 1  # 0.99
train_batch = 2560 # 2560  # 4000  # default 4000
mini_batch = 128 # default: 128
rollout_fragment_length = 2560
num_workers = 0
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

# instantiate replay and pass agent object as input argument
replay = Replay(rl_agent=agent,
                episode_length=agent.episode_length,
                # Note: saved for later when I run on several symbols.
                # Testset:
                identifier_list=['BAY', 'SAP', 'LIN', 'ALV', 'DTE'],
                random_identifier=True,
                start_date="2021-01-01",
                end_date="2021-04-30",#"2021-04-30",
                shuffle=True,
                #####
                verbose=False)

# -- Generate config file for PPO trainer.

# -- Environment Set-up.
config = {}
config["env"] = TradingEnvironment
config["env_config"] = {
    "config": {
        "replay": replay},
}
# Size of the observation space
config["env_config"]["observation_size"] = obs_size
config["env_config"]["action_size"] = action_size

# -- PPO SETUP
# Notes: 0:  use the learner GPU for inference.
config["num_workers"] = num_workers
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

config["framework"] = "tf"
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


######### EVAL
config['in_evaluation'] = True

#####
# -- Instantiate the Trainer object using above config.
trained_strategy = PPOTrainer(config=config)

#print("config")
#print(trained_strategy.config)

# -- Reload from checkpoint.
if restoring_checkpoint_path:
    trained_strategy.restore(restoring_checkpoint_path)
    print("RESTORED FROM CHECKPOINT, iterations: ", trained_strategy._iteration)

# -- Result path.



#results = []
#result_path = generate_test_result_path(symbol=replay.identifier,
#                                        strategy_name=STRATEGY_NAME)
#print("(INFO) TEST RESULTS WILL BE STORED TO: ", result_path)
###############
# -- Test loop.
# TODO: dont use config dict but traied_strategty.config!!!
print("conf keys 1: ", trained_strategy.config.keys())
print("'evaluation_config' keys:",  trained_strategy.config['evaluation_config'].keys())
print("Entire trained_strategy.config['evaluation_config']", trained_strategy.config['evaluation_config'])
print("Eval: trained_strategy.config['evaluation_config']['env_config']", trained_strategy.config['evaluation_config']['env_config'])
print("Standard: trained_strategy.config['env_config'] ", trained_strategy.config['env_config'])
eval_env_config = trained_strategy.config['evaluation_config']['env_config']

# TODO: If this does not work I have to set it before the agent is instatiated.
#trained_strategy.config['in_evaluation'] = True

print("Activated Evaluation Mode", trained_strategy.config['in_evaluation'])

# TODO: that actually doesnt make any sense.
# trained_strategy.config['evaluation_config']['model']['use_lstm'] = False
# TODO: die settings innerhalb von eval_env anschauen

# TODO
#'explore': True
###############

# Instantiate environment.
env = TradingEnvironment(eval_env_config) #config["env_config"])
# Reset env, get initial obs.
print("BEFORE ENV RESET.")
obs = env.reset()

import numpy as np
init_state = [np.zeros([lstm_cell_size], np.float32) for _ in range(2)]
state = init_state.copy()



for i in range(100):
    # compute first action based on initial state.
    action, state_out, _ = trained_strategy.compute_single_action(obs, state)
    print("a", action)
    state = state_out


print("COMPUTED SPECIAL ACTION!")


# Dict to store rewards for each test-episode.
reward_dict = {}
episode_counter = 0
episode_reward = 0

print("AFTER ENV RESET.")
# Try excet since eventually the episode start list will be other.
#try:
NUM_TEST_EPISODES = 10
while episode_counter < NUM_TEST_EPISODES:
    # Compute action.

    print("BEFORE COMPUTE FIRST ACTION.")
    # Note: `compute_action` has been deprecated. Use `Algorithm.compute_single_action()
    action = trained_strategy.compute_single_action(
        observation=obs,
        explore=False,
        # TODO: warum habe ich das nochmal gemacht?
        policy_id="default_policy"
    )
    print("action", action)

    print("AFTER COMPUTE FIRST ACTION.")
    # Send action to env and take step receiving obs, reward, done, info.
    obs, reward, done, info = env.step(action)
    # Count the reward.
    episode_reward += reward

    # If episode is done, collect stats and reset env.
    if done:
        import copy
        # -- Store results.
        print("EPISODE DONE REACHED.")
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
'''
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

'''
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
