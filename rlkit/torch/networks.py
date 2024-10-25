"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm, EnsembleLinear

def identity(x):
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    #x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x

class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            obs_mean=0.0,
            obs_std=1.0,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.obs_mean = obs_mean
        self.obs_std = obs_std

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class NormFlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *input, **kwargs):
        obs, act = input[0], input[1]
        obs = (obs - self.obs_mean.to(ptu.device)) / self.obs_std.to(ptu.device)
        flat_inputs = torch.cat([obs, act], dim=1)
        return super().forward(flat_inputs, **kwargs)

class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

class ParallelizedLayerMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            input_dim,
            output_dim,
            w_std_value=1.0,
            b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = ptu.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = ptu.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b

class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity

        self.layer_norm = layer_norm

        self.fcs = []

        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                # hidden_init(fc.W[j], w_scale)
                hidden_init(fc.W[j])
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d' % i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                ptu.orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        state_dim = inputs[0].shape[-1]

        dim = len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)

        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output

    def sample(self, *inputs):
        preds = self.forward(*inputs)

        return torch.min(preds, dim=0)[0]

    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)

# class EnsembleTransitionMLP(nn.Module):
#
#     def __init__(
#         self,
#         state_dim: int, action_dim: int, hidden_dim: int, depth:int, ensemble_size=5, mode='local', with_reward=True
#     ) -> None:
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.ensemble_size = ensemble_size
#         self.mode = mode
#         self.activation = Swish()
#         self.with_reward = with_reward
#
#         self._net = EnsembleMLP(state_dim + action_dim, hidden_dim, depth, state_dim + 1, ensemble_size)
#
#     def forward(
#         self, s: torch.Tensor, a: torch.Tensor
#     ) :
#         sa = torch.cat([s, a], dim=1)
#         return self._net(sa)
#
#     def set_select(self, indexes):
#         for layer in self._net:
#             if isinstance(layer, EnsembleLinear):
#                 layer.set_select(indexes)
#
#     def update_save(self, ):
#         for layer in self._net:
#             if isinstance(layer, EnsembleLinear):
#                 layer.update_save()



class EnsembleGaussian(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, ensemble_size=20, mode='local', with_reward=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size

        self.activation = Swish()

        module_list = []
        for i, hidden_dim in enumerate(hidden_features):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_dim, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_dim, hidden_dim, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_dim, 2 * (obs_dim + self.with_reward), ensemble_size)

        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * 1, requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * -5, requires_grad=True))

    def forward(self, obs, action):
        output = obs_action = torch.cat([obs, action], dim=-1)
        for layer in self.backbones:
            output = self.activation(layer(output))
        mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
        logstd = soft_clamp(logstd, (self.min_logstd, self.max_logstd))
        if self.mode == 'local':
            if self.with_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]
        return torch.distributions.Normal(mu, torch.exp(logstd))

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)

    def update_save(self, indexes):
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

# class EnsembleTransitionMLP(nn.Module):
#
#     def __init__(
#         self,
#         state_dim: int, action_dim: int, hidden_dim: int, depth:int, ensemble_size=5, mode='local', with_reward=True
#     ) -> None:
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.ensemble_size = ensemble_size
#         self.mode = mode
#         self.activation = Swish()
#         self.with_reward = with_reward
#
#         self._net = EnsembleMLP(state_dim + action_dim, hidden_dim, depth, state_dim + 1, ensemble_size)
#
#     def forward(
#         self, s: torch.Tensor, a: torch.Tensor
#     ) :
#         sa = torch.cat([s, a], dim=1)
#         return self._net(sa)
#
#     def set_select(self, indexes):
#         for layer in self._net:
#             if isinstance(layer, EnsembleLinear):
#                 layer.set_select(indexes)
#
#     def update_save(self, ):
#         for layer in self._net:
#             if isinstance(layer, EnsembleLinear):
#                 layer.update_save()
