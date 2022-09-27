
# Source: https://docs.ray.io/en/latest/rllib/rllib-env.html

# Note: The gym registry is not compatible with Ray. Instead, always use the
# registration flows documented above to ensure Ray workers can access the
# environment.

# TODO: test with Environment

from ray.tune.registry import register_env

def env_creator(env_config):
    return MyEnv(...)  # return an env instance

register_env("my_env", env_creator)
algo = ppo.PPO(env="my_env")