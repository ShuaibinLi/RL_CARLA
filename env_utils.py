import copy

import parl
import carla
import gym
import gym_carla
import numpy as np
from parl.utils import logger, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper
import matplotlib.pyplot as plt
from PIL import Image
from torch_base import DetectBoundingBox


class ParallelEnv(object):
    def __init__(self, env_name, xparl_addr, train_envs_params):
        parl.connect(xparl_addr)
        self.env_list = [
            CarlaRemoteEnv(env_name=env_name, params=params)
            for params in train_envs_params
        ]
        self.env_num = len(self.env_list)
        self.episode_reward_list = [0] * self.env_num
        self.episode_steps_list = [0] * self.env_num
        self._max_episode_steps = train_envs_params[0]['max_time_episode']
        self.total_steps = 0

    def reset(self):
        # print("env_utils.py:", "reset function")
        obs_list = [env.reset() for env in self.env_list]
        # print("Resetting Envs:", obs_list)
        obs_list = [obs.get() for obs in obs_list]
        # print("getting observations:", obs_list)
        self.obs_list = np.array(obs_list)
        return self.obs_list

    def step(self, action_list):
        return_list = [
            self.env_list[i].step(action_list[i]) for i in range(self.env_num)
        ]
        return_list = [return_.get() for return_ in return_list]
        return_list = np.array(return_list, dtype=object)
        self.next_obs_list = return_list[:, 0]
        self.reward_list = return_list[:, 1]
        self.done_list = return_list[:, 2]
        self.info_list = return_list[:, 3]
        return self.next_obs_list, self.reward_list, self.done_list, self.info_list

    def get_obs(self):
        for i in range(self.env_num):
            self.total_steps += 1
            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += self.reward_list[i]

            self.obs_list[i] = self.next_obs_list[i]
            if self.done_list[i] or self.episode_steps_list[
                    i] >= self._max_episode_steps:
                tensorboard.add_scalar('train/episode_reward_env{}'.format(i),
                                       self.episode_reward_list[i],
                                       self.total_steps)
                logger.info('Train env {} done, Reward: {}'.format(
                    i, self.episode_reward_list[i]))

                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0
                obs_list_i = self.env_list[i].reset()
                self.obs_list[i] = obs_list_i.get()
                self.obs_list[i] = np.array(self.obs_list[i])
        return self.obs_list


class LocalEnv(object):
    def __init__(self, env_name, params):
        # print("Local Env Called")
        self.env = gym.make(env_name, params=params)
        # print("4"*50, "env_utils.py")
        self.env = ActionMappingWrapper(self.env)
        # print("Low Bound:", self.env.low_bound)
        # print("High Bound:", self.env.high_bound)
        self._max_episode_steps = int(params['max_time_episode'])
        # print("Max Episodes:", self._max_episode_steps)
        self.obs_dim = self.env.state_space.shape[0]
        # print('State Space:', self.env.state_space)
        # print("Obs Dim:", self.obs_dim)
        self.action_dim = self.env.action_space.shape[0]
        # print('Action Space:', self.env.action_space)
        # print("Obs Dim:", self.action_dim)

    def to_bgra_array(self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def to_rgb_array(self, image):
        """Convert a CARLA raw image to a RGB numpy array."""
        array = self.to_bgra_array(image)
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def reset(self):
        # print("env_utils.py reset")
        obs, _, current_image = self.env.reset()
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            plt.imshow(numpy_rgb_image)
            plt.savefig("carla_rgb_sensor_detected/" + str(current_image.frame) + '.png')
        #     print("$" * 25, "RESET Image Name:", str(current_image.frame), "$" * 25)
        #     faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
        #     faster_rcnn_obj.detect_bounding_boxes()
        return obs

    def step(self, action):
        action_out, current_image = self.env.step(action)
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            plt.imshow(numpy_rgb_image)
            plt.savefig("carla_rgb_sensor_detected/" + str(current_image.frame) + '.png')
            # print("$" * 25, "STEP Image Name:", str(current_image.frame), "$" * 25)
            # faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
            # faster_rcnn_obj.detect_bounding_boxes()
        return action_out

@parl.remote_class(wait=False)
class CarlaRemoteEnv(object):
    def __init__(self, env_name, params):
        class ActionSpace(object):
            def __init__(self,
                         action_space=None,
                         low=None,
                         high=None,
                         shape=None,
                         n=None):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n

            def sample(self):
                return self.action_space.sample()

        self.env = gym.make(env_name, params=params)
        self.env = ActionMappingWrapper(self.env)
        self._max_episode_steps = int(params['max_time_episode'])
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)
