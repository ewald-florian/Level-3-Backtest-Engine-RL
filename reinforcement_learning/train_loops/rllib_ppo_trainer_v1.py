
import time
import json
import pprint

# TODO: Ich glaube das Problem liegt darin, dass ich Markek in Replay richtig
#  instanzieren muss damit es funktioniert... Könnte leider sein, dass die
#  anderen klassen attribute ebenfalls irgendwie probleme machen (OMS,
#  AgentTrade etc)

# rllib imports
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG

# library imports
from reinforcement_learning.environment import Environment
from reinforcement_learning.rl_agents.sample_agent import RlAgent
from replay.replay import Replay

### change name ###
'''
# TODO: put this in a function in utils
name = "test_run"
timestr = time.strftime("_%Y%m%d_%H%M%S")
result_dir = "/Users/florianewald/PycharmProjects/Level-3-Backtest-Engine-RL/reinforcement_learning/training_results"
result_file = result_dir + name + timestr + "_results.csv"
print(result_file)
'''

# Start a new instance of Ray
ray.init()

# instantiate agent
agent = RlAgent()
# instantiate replay and pass agent object as input argument
replay = Replay(rl_agent=agent)

config = PPO_DEFAULT_CONFIG
config["env"] = Environment
config["env_config"] = {
    "config": {
        "replay": replay},
}

#config["num_workers"] = 0

config["disable_env_checking"] = True


# Instantiate the Trainer object using above config.
rllib_trainer = PPOTrainer(config=config)

# print(rllib_trainer.get_policy().model.base_model.summary())

num_iterations = 1

results = []
episode_data = []
episode_json = []



for n in range(num_iterations):
    result = rllib_trainer.train()
    results.append(result)

ray.shutdown()
'''

    # store relevant metrics from the result dict to the episode dict
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
    #result_df.to_csv(result_file, index=False)
    #file_name = rllib_trainer.save(checkpoint_root)

    #print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_mean"]}')

result_df = pd.DataFrame(data=episode_data)
print(result_df.head())
'''
#result_df.to_csv(result_file, index=False)

#checkpoint_file = rllib_trainer.save()
#print(100 * '-')
#print(f"Trainer (at iteration {rllib_trainer.iteration} was saved in '{checkpoint_file}'!")