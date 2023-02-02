""" debug the result dict"""

import ray
from ray.rllib.agents.ppo import PPOTrainer
import json
import pandas as pd
from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
from reinforcement_learning.environment.tradingenvironment import TradingEnvironment
from replay_episode.replay import Replay
from utils.result_path_generator import generate_result_path

file_name = generate_result_path('tester1')

# Start a new instance of Ray
ray.init()



replay = Replay()

config = PPO_DEFAULT_CONFIG

# update config with custom env
config["env"] = TradingEnvironment
config["env_config"] = {
    "config": {
        "replay": replay},
}
#config["disable_env_checking"]=True

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


# Instantiate the Trainer object using above config.
rllib_trainer = PPOTrainer(config=config)

# print(rllib_trainer.get_policy().model.base_model.summary())

print(config)

num_iterations = 1000

results = []
episode_data = []
episode_json = []


for n in range(num_iterations):
    result = rllib_trainer.train()
    results.append(result)

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
    result_df.to_csv(result_file, index=False)
    #file_name = rllib_trainer.save(checkpoint_root)

#    print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_mean"]}')


result_df = pd.DataFrame(data=episode_data)
# print(result_df.head())

result_df.to_csv(result_file, index=False)

#checkpoint_file = rllib_trainer.save()
#print(100 * '-')
#print(f"Trainer (at iteration {rllib_trainer.iteration} was saved in '{checkpoint_file}'!")






