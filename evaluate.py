import argparse
from env_utils import LocalEnv
from parl.utils import logger, tensorboard
# from torch_base import TorchModel, TorchSAC, TorchAgent  # Choose base wrt which deep-learning framework you are using
from paddle_base import PaddleModel, PaddleSAC, PaddleAgent
from env_config import EnvConfig

EVAL_EPISODES = 3
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4


def run_episode(agent, env):
    episode_reward = 0.
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < env._max_episode_steps:
        steps += 1
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.set_dir('./{}_eval'.format(args.env))

    # env for eval
    eval_env_params = EnvConfig['test_env_params']
    eval_env = LocalEnv(args.env, eval_env_params)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    # Initialize model, algorithm, agent
    if args.framework == 'torch':
        CarlaModel, SAC, CarlaAgent = TorchModel, TorchSAC, TorchAgent
    elif args.framework == 'paddle':
        CarlaModel, SAC, CarlaAgent = PaddleModel, PaddleSAC, PaddleAgent
    model = CarlaModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = CarlaAgent(algorithm)
    # restore trained agent
    agent.restore('./{}'.format(args.restore_model))

    # Evaluate episode
    for episode in range(args.eval_episodes):
        episode_reward = run_episode(agent, eval_env)
        tensorboard.add_scalar('eval/episode_reward', episode_reward, episode)
        logger.info('Evaluation episode reward: {}'.format(episode_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument(
        '--framework',
        default='paddle',
        help='deep learning framework: torch or paddle')
    parser.add_argument(
        "--eval_episodes",
        default=10,
        type=int,
        help='max time steps to run environment')
    parser.add_argument(
        "--restore_model",
        default='model.ckpt',
        type=str,
        help='restore saved model')
    args = parser.parse_args()

    main()
