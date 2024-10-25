from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.sac.policies import VAEPolicy

EPS = 1e-10

def identity(x):
    return x

class CQLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            vae,
            target_qf1,
            target_qf2,

            raw_action,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            num_qs=2,

            min_q_version=3,
            temp=1.0,
            ratio_temp=1.0,
            tau=1.0,
            min_q_weight=1.0,
            c_min=0.0,
            tot_weight_clip=0.0,

            max_q_backup=False,
            deterministic_backup=True,
            logn=10,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,

            policy_eval_start=0,
            kl_weight=0.5,
            entropy_const=-1.,

            logsum=False,
    ):
        super().__init__()
        self.env = env

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vae = vae
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        self.policy_eval_start = policy_eval_start
        self.soft_target_tau = soft_target_tau

        self.logsum = logsum
        self.raw_action = raw_action
        self.logn = logn

        self.entropy_const = entropy_const
        self.use_automatic_entropy_tuning = False if self.entropy_const >= 0. else True
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_entropy_const = ptu.zeros(1, requires_grad=True)
            self.auto_ent_optimizer = optimizer_class(
                [self.log_entropy_const],
                lr=policy_lr,
            )

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            print("lagrange_thresh: ", lagrange_thresh)
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )
        else:
            self.log_alpha_prime = torch.zeros(1).to(ptu.device)

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vae_optimizer = optimizer_class(
            self.vae.parameters(),
            lr=policy_lr,
        )

        self.discount = discount
        print("self.discount: ", discount)
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self.eval_wandb = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

        self.num_qs = num_qs

        self.temp = temp
        self.ratio_temp = ratio_temp
        self.tau = tau
        self.kl_weight = kl_weight

        self.c_min = c_min
        self.tot_weight_clip = tot_weight_clip
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight
        print("self.min_q_version: ", min_q_version)
        print("self.min_q_weight: ", min_q_weight)

        self.softmax = torch.nn.Softmax(dim=1)

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        self.discrete = False

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        if isinstance(network, VAEPolicy):
            new_obs_actions = network.decode(obs_temp)
            new_obs_log_pi = network.log_prob(obs_temp, new_obs_actions)
        else:
            new_obs_actions, _, _, new_obs_log_pi, *_ = network(
                obs_temp, reparameterize=False, return_log_prob=True,
            )
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions

    def train_from_torch(self, batch):
        self._current_epoch += 1

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        is_weight = batch['is_weight']

        """
        VAE Loss
        """

        recon, mean, std = self.vae(obs, actions)
        recon_loss = (recon - actions) ** 2
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2))
        vae_loss = recon_loss.mean() + self.kl_weight * kl_loss.mean()

        loss = vae_loss.mean()

        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()

        """
        Policy and Entropy Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            entropy_const_loss = (self.log_entropy_const * (-log_pi - self.target_entropy).detach()).mean()
            self.auto_ent_optimizer.zero_grad()
            entropy_const_loss.backward()
            self.auto_ent_optimizer.step()
            entropy_const = self.log_entropy_const.exp()
        else:
            entropy_const = self.entropy_const
            entropy_const_loss = 0

        if self.num_qs == 1:
            q_new_actions = self.qf1(obs, new_obs_actions)
        else:
            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions),
            )

        policy_loss = (entropy_const * log_pi - q_new_actions).mean()

        if self._current_epoch < self.policy_eval_start:
            policy_log_prob = self.policy.log_prob(obs, actions)
            policy_loss = (entropy_const * log_pi - policy_log_prob).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        if self.num_qs > 1:
            q2_pred = self.qf2(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=False, return_log_prob=True,
        )
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
            obs, reparameterize=False, return_log_prob=True,
        )

        if not self.max_q_backup:
            if self.num_qs == 1:
                target_q_values = self.target_qf1(next_obs, new_next_actions)
            else:
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),
                )

            if not self.deterministic_backup:
                target_q_values = target_q_values - entropy_const * new_log_pi

        if self.max_q_backup:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[
                0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[
                0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        clipped_is_weight = is_weight.clamp(min=self.c_min)
        bellman_qf1_loss = (((q1_pred - q_target) ** 2) * clipped_is_weight).mean()
        bellman_qf2_loss = (((q2_pred - q_target) ** 2) * clipped_is_weight).mean()

        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random,
                                                                     network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random,
                                                                        network=self.policy)

        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1,1)
        if new_curr_actions_tensor.is_cuda:
            random_actions_tensor = random_actions_tensor.cuda()

        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)

        random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])

        with torch.no_grad():
            if self.logsum:
                pi_log_betas = self.vae.log_prob(obs, curr_actions_tensor, logsum=self.logsum, comp_raw_action = self.raw_action, logn=self.logn).view(obs.shape[0], -1)
            else:
                pi_log_betas = self.vae.log_prob(obs, curr_actions_tensor, comp_raw_action = self.raw_action).view(obs.shape[0], -1)

            threshold = random_density * self.tau

            log_pi_ratio = (pi_log_betas - threshold).clamp(min=0,)

            pi_weight = torch.exp(-log_pi_ratio / self.ratio_temp).detach().mean(dim=-1, keepdim=True)

        if self.min_q_version == 3:
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(),
                 q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(),
                 q2_curr_actions - curr_log_pis.detach()], 1
            )
        elif self.min_q_version == 2:
            cat_q1 = torch.cat(
                [q1_rand, q1_next_actions, q1_curr_actions, q1_pred.unsqueeze(1)], 1
            )
            cat_q2 = torch.cat(
                [q2_rand, q2_next_actions, q2_curr_actions, q2_pred.unsqueeze(1)], 1
            )
        else:
            raise NotImplementedError

        logsum_ratio_q1 = pi_weight
        min_q_ratio = (is_weight * pi_weight).clamp(self.tot_weight_clip)

        min_qf1_loss = (min_q_ratio * (torch.logsumexp(cat_q1 / self.temp, dim=1, ) - q1_pred)).mean() * self.min_q_weight * self.temp
        min_qf2_loss = (min_q_ratio * (torch.logsumexp(cat_q2 / self.temp, dim=1, ) - q2_pred)).mean() * self.min_q_weight * self.temp

        alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
        if self.with_lagrange:
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = bellman_qf1_loss + min_qf1_loss
        qf2_loss = bellman_qf2_loss + min_qf2_loss

        """
        Update networks
        """
        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        if self.num_qs > 1:
            self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            self.qf2_optimizer.step()

        """
        Soft Updates
        """
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        if self.num_qs > 1:
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics.update(create_stats_ordered_dict(
                'IS Weight',
                ptu.get_numpy(is_weight),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Pi log beta',
                ptu.get_numpy(pi_log_betas),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Pi weight',
                ptu.get_numpy(pi_weight),
            ))

            self.eval_statistics['Ratio Threshold'] = threshold

            self.eval_statistics['Bellman QF1 Loss'] = np.mean(ptu.get_numpy(bellman_qf1_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
            self.eval_statistics['Total QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            if self.num_qs > 1:
                self.eval_statistics['Bellman QF2 Loss'] = np.mean(ptu.get_numpy(bellman_qf1_loss))
                self.eval_statistics['min QF2 Loss'] = np.mean(ptu.get_numpy(min_qf2_loss))
                self.eval_statistics['Total QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))


            self.eval_statistics.update(create_stats_ordered_dict(
                'logsum_ratio',
                ptu.get_numpy(logsum_ratio_q1),
            ))

            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 curr_actions values',
                    ptu.get_numpy(q1_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 curr_actions values',
                    ptu.get_numpy(q2_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 random values',
                    ptu.get_numpy(q1_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 random values',
                    ptu.get_numpy(q2_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 next_actions values',
                    ptu.get_numpy(q1_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 next_actions values',
                    ptu.get_numpy(q2_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'actions',
                    ptu.get_numpy(actions)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'rewards',
                    ptu.get_numpy(rewards)
                ))

            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps

            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            if self.num_qs > 1:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['target_entropy'] = self.target_entropy
                self.eval_statistics['Entropy Constant'] = entropy_const.item()
                self.eval_statistics['Auto Entropy Loss'] = entropy_const_loss.item()

            if self.with_lagrange:
                self.eval_statistics['Alpha_prime'] = alpha_prime.item()
                self.eval_statistics['min_q1_loss'] = ptu.get_numpy(min_qf1_loss).mean()
                self.eval_statistics['min_q2_loss'] = ptu.get_numpy(min_qf2_loss).mean()
                self.eval_statistics['threshold action gap'] = self.target_action_gap
                self.eval_statistics['alpha prime loss'] = alpha_prime_loss.item()

        self._n_train_steps_total += 1


    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.vae,
            self.target_qf1,
            self.target_qf2,
        ]
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            vae=self.vae,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )

    def set_snapshot(self, snapshot):
        self.policy = snapshot['policy']
        self.qf1 = snapshot['qf1']
        self.qf2 = snapshot['qf2']
        self.vae = snapshot['vae']
        self.target_qf1 = snapshot['target_qf1']
        self.target_qf2 = snapshot['target_qf2']
