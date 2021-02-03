import parl
import paddle
import numpy as np


class CarlaAgent(parl.Agent):
    def __init__(self, algorithm):
        super(CarlaAgent, self).__init__(algorithm)

        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1).astype(np.float32))
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def sample(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1).astype(np.float32))
        action, _ = self.alg.sample(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs)
        action = paddle.to_tensor(action)
        reward = paddle.to_tensor(reward)
        next_obs = paddle.to_tensor(next_obs)
        terminal = paddle.to_tensor(terminal)
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs, terminal)
        return critic_loss, actor_loss
