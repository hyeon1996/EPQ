from collections import deque, OrderedDict

import rlkit.torch.pytorch_util as ptu
import matplotlib.pyplot as plt

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.rollout_functions import rollout, batcheval_rollout, multitask_rollout, function_rollout
from rlkit.samplers.data_collector.base import PathCollector
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# 신경망 구조 정의
class DimReduction(nn.Module):
    def __init__(self):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(3, 1)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # 출력층 후 Tanh 활성화 함수 적용
        return x

class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            sparse_reward=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._sparse_reward = sparse_reward

    def update_policy(self, new_policy):
        self._policy = new_policy
    
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            policy_fn=None,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len

            ## Used to sparsify reward
            if self._sparse_reward:
                random_noise = np.random.normal(size=path['rewards'].shape)
                path['rewards'] = path['rewards'] + 1.0*random_noise

            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )

    def set_snapshot(self, snapshot):
        self._env = snapshot['env']
        self._policy = snapshot['policy']

class BatchEvalPathCollector:
    def __init__(
            self,
            path,
            env,
            policy,
            qf1,
            qf2,
            init_obs,
            init_index,
            bat_init_actions,
            gamma=0.99,
            N_episodes=10,
            max_path_length_this_loop=1000,
    ):
        self._env = env
        self._policy = policy
        self._qf1 = qf1
        self._qf2 = qf2
        self._path = path

        self.init_obs = ptu.from_numpy(init_obs)
        self.bat_init_actions = ptu.from_numpy(bat_init_actions)
        self.init_index = init_index

        self.gamma = gamma
        self.N_episodes = N_episodes

        self.max_path_length_this_loop = max_path_length_this_loop

    def update_policy(self, new_policy, qf1, qf2):
        self._policy = new_policy
        self._qf1 = qf1
        self._qf2 = qf2

    def plot_the_difference(self, type):

        if type =='batch':
            init_actions = self.bat_init_actions
        elif type == 'policy':
            init_actions, *_ = self._policy(self.init_obs)

        bat_q_vals = ptu.get_numpy(torch.min(self._qf1(self.init_obs, init_actions),
                                             self._qf2(self.init_obs, init_actions)))

        init_actions = ptu.get_numpy(init_actions)

        bat_returns, q_vals, long_bat_returns = batcheval_rollout(
            env = self._env,
            agent = self._policy,
            qf1 = self._qf1,
            qf2 = self._qf2,
            init_obs=self.init_obs,
            init_actions=init_actions,
            gamma=self.gamma,
            N_episodes=self.N_episodes,
            max_path_length=self.max_path_length_this_loop,
        )

        bat_diff = bat_q_vals.squeeze() - bat_returns
        bat_var = bat_diff ** 2

        tot_bat_diff = q_vals - long_bat_returns
        tot_bat_var = tot_bat_diff ** 2

        return bat_var.mean(), bat_diff.mean(), bat_q_vals.mean(), bat_returns.mean(), tot_bat_var.mean(), tot_bat_diff.mean()

class CustomMDPPathCollector(PathCollector):
    def __init__(
        self,
        env,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self, policy_fn, max_path_length, 
            num_steps, discard_incomplete_paths
        ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = function_rollout(
                self._env,
                policy_fn,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)

        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
        )

    def set_snapshot(self, snapshot):
        self._env = snapshot['env']


class GoalConditionedPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                return_dict_obs=True,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )

    def set_snapshot(self, snapshot):
        self._env = snapshot['env']
        self._policy = snapshot['policy']
        self._observation_key = snapshot['observation_key']
        self._desired_goal_key = snapshot['desired_goal_key']
