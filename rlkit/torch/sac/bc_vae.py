from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class BCTrainer(TorchTrainer):
    """
    Trainer for Behavior Cloning
    Policy is trained by maximizing log likelihood of actions in a given dataset.
    Q function is trained by SARSA
    """
    def __init__(
            self,
            env,
            policy,
            qf,
            target_qf,

            kl_weight=0.5,
            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-4,
            qf_lr=1e-4,
            optimizer_class=optim.Adam,

            soft_target_tau=5e-3,
            target_update_period=2,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.kl_weight = kl_weight
        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.discrete = False

    def train_from_torch(self, batch):

        obs = batch['obs']
        actions = batch['actions']
        next_obs = batch['next_obs']
        next_actions = batch['next_actions']
        rewards = batch['rewards']
        terminals = batch['terminals']

        """
        VAE Loss
        """

        recon, mean, std = self.policy(obs, actions)
        recon_loss = (recon - actions) ** 2
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2))
        policy_loss = recon_loss.mean() + self.kl_weight * kl_loss.mean()

        """
        QF Loss
        """

        q_pred = self.qf(obs, actions)

        target_q_values = self.target_qf(next_obs, next_actions)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()


        """
        Soft Updates
        """

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf, self.target_qf, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_qf,
        ]

    def get_snapshot(self):
        return dict(
            vae=self.policy,
            qf=self.qf,
            target_qf=self.target_qf,
        )

    def set_snapshot(self, snapshot):
        self.policy = snapshot['vae']
        self.qf = snapshot['qf']
        self.target_qf = snapshot['target_qf']