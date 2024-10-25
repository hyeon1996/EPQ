import numpy as np
import torch
import torch.nn as nn
# from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import Normal, TanhNormal
from rlkit.torch.networks import Mlp
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0

def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)

def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    x = low + 0.5 * (high - low) * (x + 1)
    return x

class TanhGaussianPolicy(Mlp, ExplorationPolicy):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            log_min=LOG_SIG_MIN,
            log_max=LOG_SIG_MAX,
            max_action=1.0,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_min = log_min
        self.log_max = log_max
        self.log_std = None
        self.std = std
        self.max_action = max_action
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert self.log_min <= self.log_std <= self.log_max

    def get_action(self, obs_np, deterministic=False):
        if isinstance(obs_np, tuple):
            obs_np = obs_np[0]
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def log_prob(self, obs, actions, std_scale=1.0, return_dist=False):
        actions = actions / self.max_action
        raw_actions = atanh(actions)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std,  self.log_min,  self.log_max)
            std = torch.exp(log_std) * std_scale
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)

        if return_dist:
            return log_prob.sum(-1), mean, std
        else:
            return log_prob.sum(-1)

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std,  self.log_min,  self.log_max)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean) * self.max_action
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample() * self.max_action
                else:
                    action = tanh_normal.sample() * self.max_action

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class GaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            log_min=LOG_SIG_MIN,
            log_max=LOG_SIG_MAX,
            output_activation=nn.Tanh(),
            obs_mean=0.0,
            obs_std=1.0,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * 2,
            output_activation=output_activation,
            init_w=init_w,
            **kwargs
        )
        self.log_min = log_min
        self.log_max = log_max
        self.std = std

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def log_prob(self, obs, actions, std=None, std_scale=1.0, return_std=False):

        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean, log_std = self.output_activation(self.last_fc(h)).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, (self.log_min, self.log_max))
        if std is not None:
            std = torch.ones_like(mean) * std
        else:
            std = log_std.exp() * std_scale

        normal = Normal(mean, std)
        log_prob = normal.log_prob(value=actions)

        if return_std:
            return log_prob.sum(-1), std
        else:
            return log_prob.sum(-1)

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            std=None,
            std_scale=1.0,
            return_std=False,
            minimum_std=None,
            std_clamp=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean, log_std = self.output_activation(self.last_fc(h)).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, (self.log_min, self.log_max))
        if std is not None:
            std = torch.ones_like(mean) * std
        else:
            std = log_std.exp() * std_scale
            if std_clamp:
                std = torch.clamp(std, max=0.5)

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            if return_log_prob:

                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()
                log_prob = normal.log_prob(action)
                log_prob = log_prob.sum(dim=1, keepdim=True)
                entropy = -log_prob
            else:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

class VAEPolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        latent_dim,
        noise_clip=0.5,
        init_w=1e-3,
        **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.noise_clip = noise_clip
        self.latent_dim = latent_dim

        self.e1 = torch.nn.Linear(obs_dim + action_dim, 512)
        self.e2 = torch.nn.Linear(512, 512)

        self.mean = torch.nn.Linear(512, self.latent_dim)
        self.log_std = torch.nn.Linear(512, self.latent_dim)

        self.d1 = torch.nn.Linear(obs_dim + self.latent_dim, 512)
        self.d2 = torch.nn.Linear(512, 512)
        self.d3 = torch.nn.Linear(512, action_dim)

        self.max_action = 1.0
        self.latent_dim = latent_dim

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np)[0]

    def forward(self, state, action=None, deterministic=True):
        if action is None:
            action = self.decode(state)
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)

        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn(state.size(0), self.latent_dim).to(state.device)
            z = torch.clamp(z, -self.noise_clip, self.noise_clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))

    def decode_multiple(self, state, z=None, num_decode=10):
        if z is None:
            z = torch.randn(state.size(0), num_decode, self.latent_dim).to(state.device)
            z = torch.clamp(z, -self.noise_clip, self.noise_clip)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(1).expand(-1, num_decode, -1), z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)

    def log_prob(self, state, action_hat, logn=1, comp_raw_action=False, logsum=False):

        action_shape = action_hat.shape[0]
        obs_shape = state.shape[0]
        num_decode = int(action_shape / obs_shape)

        actions, raw_actions = self.decode_multiple(state, num_decode=num_decode)

        if logsum:
            obs_temp = state.unsqueeze(1).repeat(1, num_decode, 1).view(state.shape[0] * num_decode,
                                                                           state.shape[1])

            # torch.Size([5120, N, 3])
            actions, raw_actions = self.decode_multiple(obs_temp, num_decode=logn)
            # actions : BN x N x A, action_hat : BN x 1 x A, log_prob : BN x 1
            # torch.Size([5120, 1, 3])
            action_hat = action_hat.unsqueeze(1)
            if comp_raw_action:
                actions = raw_actions
                action_hat = atanh(action_hat)

            # torch.Size([5120, 1]) # temp = sigma = 1
            log_prob = torch.logsumexp(-((actions - action_hat) ** 2 / 1.0).sum(dim=-1), dim=1, keepdim=True)# - np.log(logn)
        else:
            actions = actions.view(-1, action_hat.shape[-1])
            if comp_raw_action:
                actions = raw_actions.view(-1, action_hat.shape[-1])
                action_hat = atanh(action_hat)
            log_prob = -F.mse_loss(actions, action_hat, reduction='none').sum(dim=-1) # |A| * (0 ~ 4) * ratio_sacle

        return log_prob

    def log_prob2(self, state, action, action_hat, comp_raw_action=False):

        action_shape = action_hat.shape[0]
        obs_shape = state.shape[0]
        num_decode = int(action_shape / obs_shape)

        actions = action.view(-1, action_hat.shape[-1]).repeat(num_decode, 1)

        if comp_raw_action:
            actions = atanh(actions)
            action_hat = atanh(action_hat)

        log_prob = -F.mse_loss(actions, action_hat, reduction='none') # + KL (0 ~ 4*A)

        return log_prob.sum(dim=-1)