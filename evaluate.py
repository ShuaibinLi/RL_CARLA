import gym
import numpy as np
import parl
import argparse
import carla
import gym_carla
from parl.utils import logger, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper
from carla_model import CarlaModel
from carla_agent import CarlaAgent
from sac import SAC


GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
_max_episode_steps = 250


def run_evaluate_episodes(agent, env):
    episode_reward = 0.
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < _max_episode_steps:
        steps += 1
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")
    logger.set_dir('./{}_eval_{}'.format(args.env, args.seed))

    # env for eval
    params = {
        'obs_size': (160, 100),  # screen size of cv2 window
        'dt': 0.025,  # time interval between two frames
        'ego_vehicle_filter':
        'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2027,  # connection port
        'task_mode':
        'Lane',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'test',
        'max_time_episode': 250,  # maximum timesteps per episode
        'desired_speed': 15,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }
    eval_env = gym.make('carla-v0', params=params)
    eval_env.seed(args.seed)
    eval_env = ActionMappingWrapper(eval_env)

    obs_dim = eval_env.state_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    # Initialize model, algorithm
    model = CarlaModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = CarlaAgent(algorithm)
    # Restore agent
    agent.restore('./model.ckpt')

    # Evaluate episode
    for episode in range(args.evaluate_episodes):
        episode_reward = run_evaluate_episodes(agent, eval_env)
        tensorboard.add_scalar('eval/episode_reward', episode_reward, episode)
        logger.info('Evaluation episode reward: {}'.format(episode_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument("--task_mode", default='Lane', help='mode of the task')
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help='sets carla env seed for evaluation')
    parser.add_argument(
        "--evaluate_episodes",
        default=1e4,
        type=int,
        help='max time steps to run environment')
    args = parser.parse_args()

    main()
