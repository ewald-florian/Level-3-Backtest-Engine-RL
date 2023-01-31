"""
- Load some weights (it does not matter if they are already optimized)
- Set up Replay with
        - The test period
        - The correct invervals
        - The correct identifier

"""


import ray
from ray.rllib.agents.ppo import PPOTrainer
# TODO: gibt es nicht mehr.
#from ray.rllib.algorithms.algorithm import Algorithm

from reinforcement_learning.environment.tradingenvironment import \
    TradingEnvironment
from reinforcement_learning.agent_prototypes.archieve.twap_incentive_agent import \
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
#checkpoint_file = '/Users/florianewald/ray_results/PPOTrainer_TradingEnvironment_2022-12-28_13-04-25c1c62q8_/checkpoint_000001/checkpoint-1'
checkpoint_file = '/Users/florianewald/ray_results/PPOTrainer_TradingEnvironment_2023-01-24_23-08-34ii7rsr2k/checkpoint_000003/checkpoint-3'

# -- Create config dict in order to set up the PPO trainer
# Agent.
agent = TwapIncentiveAgent(verbose=True)
# Replay.
replay = Replay(
    rl_agent=agent,
    episode_length="10s",
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
config["env_config"]["observation_size"] = 34
config["env_config"]["action_size"] = 17
config["num_workers"] = 0
# TODO: enable env checking and check the environment.
#config["disable_env_checking"] = True
#########
# Initialize PPO.
trained_policy = PPOTrainer(config=config)
print('step 1 done')
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

