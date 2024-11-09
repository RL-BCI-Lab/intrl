
"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from typing import List, Type, Union
from pdb import set_trace

import torch as th
from torch import nn

from stable_baselines3.common.utils import get_device


def compare_models(model_1, model_2):
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if th.all(th.isclose(key_item_1[1], key_item_2[1])):
            print(f"Keys {key_item_1[0]} matched")
        else:
            continue
        

def reset_parameters(module):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def dryrun_module(module, obs, upto=None):
    if upto is not None:
        module = module[:upto]

    with th.no_grad():         
        return module(obs)


class MetaDataModule(nn.Module):
    """ Torch layer subclass for storing meta data for logging

        Attributes:
            metadata: Metadata compute for a single forward pass. This data will be 
                overwritten each forward pass.
    """
    def __init__(self):
        super().__init__()
        self._metadata = {}
        
    @property
    def metadata(self, *args, **kwargs):
        return self._metadata


class PNorm(MetaDataModule):
    """
        L. F. P. Cattelan and D. Silva, “Improving selective classification performance 
        of deep neural networks through post-hoc logit normalization and temperature scaling.”
    """
    def __init__(
        self, 
        p: int = 2,
        center: bool = False,
        estimate_temperature: bool =  False,  
        eps:float = 1e-12
    ):
        super().__init__()
        self.p = p
        self.center = center
        self.estimate_temperature = estimate_temperature
        self.eps = eps
        
    def forward(self, x):
        self.metadata['logits'] = x.detach()
        x = self.centralize(x) if self.center else x
        x = x/self.p_norm(x)
        self.metadata['logits_norm'] = x.detach()
        pred_tau = x.mean().detach() if self.estimate_temperature else 1
        self.metadata['pred_tau'] = pred_tau
        return x/pred_tau

    def centralize(self, x: th.tensor):
        return x - x.mean(-1).view(-1,1)

    def p_norm(self, x: th.tensor,):
        if self.p == 0: 
            return th.ones(x.size(0), 1, device=x.device)
        else: 
            return x.norm(p=self.p, dim=-1).clamp_min(self.eps).view(-1,1)


class StandardNorm(MetaDataModule):
    def __init__(self, scale: float = 1, shift: float = 0, dim: int = 1, eps: float = 1e-12):
        super().__init__()
        self.m = scale
        self.b = shift
        self.dim = dim
        self.eps = eps

    def forward(self, x: th.tensor):
        self.fit(x)
        self.metadata['original_logits'] = x.detach()
        self.metadata['logit_mu'] = self.mu.detach()
        self.metadata['logit_std'] = self.std.detach()
        return self.transform(x)

    def fit(self, x: th.tensor):
        self.mu = x.mean(dim=self.dim)
        self.std = x.std(dim=self.dim) + th.tensor(1e-11)
        if len(self.mu.shape) != 0:
            self.mu = self.mu.view(-1, 1) if len(self.mu) == len(x) else self.mu
        if len(self.std.shape) != 0:
            self.std = self.std.view(-1, 1) if len(self.std) == len(x) else self.std

    def transform(self, x: th.tensor):
        return self.m * ( ( x - self.mu ) / self.std ) + self.b
    
    def inverse_transform(self, y: th.tensor):
        return (y / self.m - self.b) * self.std + self.mu
    

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    device: Union[th.device, str] = "auto",
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net 
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function to use after each layer.
    :param squash_output: Whether to squash the output using the provided
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """
    device = get_device(device)
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(activation_fn())
        
    return nn.Sequential(*modules).to(device)

        
# class RLExtractor(nn.Module):
#     def __init__(
#         self,
#         value_net=None,
#         value_net_kwargs=None,
#         pi_net=None,
#         pi_net_kwargs=None,
#         device: Union[th.device, str] = "auto",
#     ) -> None:

#         if not (value_net and pi_net):
#             msg = "Must pass at least one of the following arguments: `value_net` or `pi_net`"
#             raise ValueError(msg)
#         super().__init__()
#         device = get_device(device)
        
#         value_net_kwargs = value_net_kwargs or {}
#         pi_net_kwargs = pi_net_kwargs or {}
        
#         self.value_net = value_net(**value_net_kwargs) if value_net else nn.Identity()
#         self.pi_net = pi_net(**pi_net_kwargs) if pi_net else nn.Identity()
        
#         # TODO: Get last layer output, we dont know input and laster layer might not have 
#         #       output dims like CNN
#         # Save dim, used to create the distributions
#         self.latent_dim_pi = None
#         self.latent_dim_vf = None


#         self.value_net.to(device)
#         self.pi_net.to(device)

#     def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         :return: latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         return self.forward_actor(features), self.forward_critic(features)

#     def forward_actor(self, features: th.Tensor) -> th.Tensor:
#         return self.policy_net(features)

#     def forward_critic(self, features: th.Tensor) -> th.Tensor:
#         return self.value_net(features)
    
