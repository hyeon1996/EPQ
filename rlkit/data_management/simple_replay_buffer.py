from collections import OrderedDict

import numpy as np
import torch
from scipy import special

from rlkit.data_management.replay_buffer import ReplayBuffer
import rlkit.torch.pytorch_util as ptu


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        self._is_weight = None

        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def add_sample_only(self, observation, action, reward, next_observation, terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size, prob=None, get_idx=False):

        if prob is None:
            indices = np.random.choice(self._size, batch_size, replace=False)
        else:
            indices = np.random.choice(self._size, batch_size, replace=True, p=prob)

        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            is_weight=self._is_weight[indices],
        )

        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]

        if get_idx == True:
            return batch, indices
        else:
            return batch

    def calculate_gamma_return(self, gamma=0.99):
        gam_return = np.zeros_like(self._rewards)
        pre_return = 0
        for i in reversed(range(self._size)):
            gam_return[i] = self._rewards[i] + gamma * pre_return * (1 - self._terminals[i])
            pre_return = gam_return[i]

        return gam_return

    def init_qf(self, qf=None, gamma=0.99):
        if qf is None:
            self._q_curr = self.calculate_gamma_return(gamma=gamma)
        else:
            self._q_curr = np.zeros_like(self._rewards)
            for i in range(0, self._size, 10000):
                end_idx = min(self._size, i + 10000)

                bat_s = ptu.from_numpy(self._observations[i:end_idx])
                bat_a = ptu.from_numpy(self._actions[i:end_idx])

                self._q_curr[i:end_idx] = ptu.get_numpy(qf(bat_s, bat_a))

        return self._q_curr

    def calculate_is_weight(self, cluster_idx_list, temp=1.0):

        q_currs = self._q_curr / temp

        self._is_weight = np.zeros_like(self._rewards)

        for i in range(0, self._size, 10000):
            end_idx = min(self._size, i + 10000)
            index_set = np.arange(i, end_idx)

            q_curr = q_currs[index_set]

            logsum_batch = np.array([special.logsumexp(
                (q_currs[cluster_idx_list[index_set[i]]]), axis=0) for i in
                range(len(index_set))])

            diff = (q_curr - logsum_batch)
            is_weight = np.exp(diff)

            self._is_weight[i:end_idx] = is_weight

        return q_currs

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])