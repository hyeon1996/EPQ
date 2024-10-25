from collections import OrderedDict
import numpy as np
from gym.spaces import Discrete
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.envs.env_utils import get_dim
import rlkit.torch.pytorch_util as ptu


class BatchReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        env_info_sizes=None,
    ):
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        self._observation_dim = get_dim(self._ob_space)
        self._action_dim = get_dim(self._action_space)
        self._max_replay_buffer_size = max_replay_buffer_size

        self._obs = np.zeros((max_replay_buffer_size, self._observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._next_actions = np.zeros((max_replay_buffer_size, self._action_dim))

        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._q_curr = None

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._top = 0
        self._size = 0

    def add_sample(self, obs, action, reward, next_obs, next_action, terminal, env_info, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_next_action = np.zeros(self._action_dim)
            new_action[action] = 1
            new_next_action[next_action] = 1
        else:
            new_action = action
            new_next_action = next_action

        self._obs[self._top] = obs
        self._actions[self._top] = new_action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_obs
        self._next_actions[self._top] = new_next_action

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def add_sample_only(self, obs, action, reward, next_obs, next_action, terminal):
        self._obs[self._top] = obs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_obs
        self._next_actions[self._top] = next_action
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
            # p = 1-D array-like
            # The probabilities associated with each entry in a.
            # If not given, the sample assumes a uniform distribution over all entries in a.
            indices = np.random.choice(self._size, batch_size, replace=True, p=prob)
        batch = dict(
            obs=self._obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_obs=self._next_obs[indices],
            next_actions=self._next_actions[indices],
            gamma = self._gamma[indices]
        )
        if self._q_curr is not None:
            batch['q_curr'] = self._q_curr[indices]

        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]

        if get_idx == True:
            return batch, indices
        else:
            return batch

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

    def init_qf(self, qf):
        self._q_curr = np.zeros_like(self._rewards)
        for i in range(0, self._size, 10000):
            end_idx = min(self._size, i + 10000)

            bat_s = ptu.from_numpy(self._obs[i:end_idx])
            bat_a = ptu.from_numpy(self._actions[i:end_idx])

            self._q_curr[i:end_idx] = ptu.get_numpy(qf(bat_s, bat_a))

    def calculate_gamma_return(self, gamma=0.99):
        gam_return = np.zeros_like(self._rewards)
        pre_return = 0
        for i in reversed(range(self._size)):
            gam_return[i] = self._rewards[i] + gamma * pre_return * (1 - self._terminals[i])
            pre_return = gam_return[i]

        self._gamma = gam_return