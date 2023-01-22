"""
Use the trained policy to compute actions i a test loop.
Load a trained agent from checkpoints, run it for a number
of test-episodes and receive the rewards as result.

Notes:
1. The set-up of the environment requires to initiate the
respective agent, just as during training. The only difference
is that the agent now receives the action directly from the
policy-call and not through the rllib train loop.

2. Replay must be initialized the same way as during training,
the only difference ist that the time-period will be set to the
test set.

3. Checkpoints can be loaded from checkpoint files when the train
loop used checkpoint_file = some_trainer.save()
"""

# TODO: save test results to an automatically generated file.
__author__ = "florian"


import ray
import os
import numpy as np
import gym
from gym import spaces
from ray.rllib.agents.ppo import PPOTrainer
# TODO: gibt es nicht mehr.
#from ray.rllib.algorithms.algorithm import Algorithm

from reinforcement_learning.environment.tradingenvironment import \
    TradingEnvironment
from reinforcement_learning.agent_prototypes.twap_incentive_agent import \
    TwapIncentiveAgent
from replay_episode.replay import Replay
from utils.episode_stats_path_generator import generate_episode_stats_path
from reinforcement_learning.environment.episode_stats import EpisodeStats

# TODO: check if this works as planed since it was originally designed for
#  training process not testing. PROBLEM: LISTE IST LEER...
# generate json file to store episode statistics.
stats_path = generate_episode_stats_path(name='test_loop')
EpisodeStats(stats_path)
print("EP_STATS_FILE:", EpisodeStats.path_name)

# Load from checkpoint of trained agent.
checkpoint_file = '/Users/florianewald/ray_results/PPOTrainer_TradingEnvironment_2022-12-28_13-04-25c1c62q8_/checkpoint_000001/checkpoint-1'

# Agent.
agent = TwapIncentiveAgent(verbose=True)
# Replay.
replay = Replay(
    rl_agent=agent,
    episode_length="1m",
    start_date="2021-02-01",
    end_date="2021-02-10",
    identifier_list=None
)
# Env config.
#env_config = {}
#env_config["config"] = {"replay_episode": replay}
#env_config["observation_size"] = 33
#env_config["action_size"] = 13

#########
config = {}
config["env"] = TradingEnvironment
config["env_config"] = {
    "config": {
        "replay_episode": replay},
}
# Size of the observation space
config["env_config"]["observation_size"] = 33
config["env_config"]["action_size"] = 13
config["num_workers"] = 0
# TODO: enable env checking and check the environment.
config["disable_env_checking"] = True
#########
# Initialize PPO.
trained_policy = PPOTrainer(config=config)
# Restore policy from checkpoints.
trained_policy.restore(checkpoint_file)
print("restored trainer:", trained_policy)

# Instantiate environment.
env = TradingEnvironment(config["env_config"])
# Reset env, get initial obs.
obs = env.reset()

# Reset counters.
num_episodes = 0
episode_reward = 0.0
# Dict to store rewards for each test-episode.
reward_dict = {}

# How many episodes should be tested.
num_test_episodes = 10

while num_episodes < num_test_episodes:
    # Compute action.
    action = trained_policy.compute_single_action(
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

# Shut down.
trained_policy.stop()
ray.shutdown()

