from gym.envs.registration import register
# print("3"*50, "gym_carla/__init__.py from env_utils.py import gym_carla")
register(
    id='carla-v0',
    entry_point='gym_carla.envs:CarlaEnv',
)
