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
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pdb import set_trace

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.distributions import (
    Distribution,
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import StateDependentNoiseDistribution


# NOTE: Device is Dynamically set based on self.parameters. First parameters would be
# the feature extractor if it is not Flatten
class ActorCriticPolicy(BasePolicy):
    """ Modified version of SB3  ActorCriticPolicy that allows for dist and mlp extractor
        to be explicitly specified. 

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param mlp_extractor_class: The MlpExtactor class containing policy and value networks
    :param mlp_extractor_kwargs: The MlpExtactor class kwargs for building an instance
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        mlp_extractor_class: Type[MlpExtractor], # NEW
        dist_class: Type[Distribution], # NEW
        mlp_extractor_kwargs: Dict = None, # NEW
        dist_kwargs: Dict = None, # NEW
        logit_norm_class: Type[nn.Module] = None,
        logit_norm_kwargs: Dict = None,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # NOTE: Seems unneeded as default value is smaller and can ruin training reproducibility 
        # if optimizer_kwargs is None:
        #     optimizer_kwargs = {}
        #     # Small values to avoid NaN in Adam optimizer
        #     if optimizer_class == th.optim.Adam:
        #         optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        self.mlp_extractor_class = mlp_extractor_class
        self.mlp_extractor_kwargs = mlp_extractor_kwargs or {}
        self.logit_norm_class = logit_norm_class
        self.logit_norm_kwargs = logit_norm_kwargs or {}
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()
            
        self.log_std_init = log_std_init
        self.dist_class = dist_class 
        self.dist_kwargs = dist_kwargs or {}
        self.action_dist = self.dist_class(**self.dist_kwargs)
        # Continuous action space related params for gSDE distribution
        # squash_output=True is only available when using gSDE (use_sde=True) issue is 
        # avoided by making use_sde true if StateDependentNoiseDistribution is used.
        # use_sde is passed to make_proba_distribution and returns StateDependentNoiseDistribution,
        # Thus if action dist is StateDependentNoiseDistribution then use_sde would have to been true
        # in the original sb3 ActorCritic code.
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.use_sde = True
            # TODO: Use what was passed to StateDependentNoiseDistribution for consistency?
            squash_output = True if self.action_dist.bijector is not None else False
            assert squash_output == self.squash_output, "Passed two different values to squash_output"
            self.use_expln = self.action_dist.use_expln
            self.full_std = self.action_dist.full_std
            assert self.action_dist.learn_features == False, "StateDependentNoiseDistribution must have learn_features set to false"
            # assert not (self.squash_output and not self.use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        else:
            self.use_sde = False
            self.use_expln = False
            self.full_std = True            

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                mlp_extractor_class=self.mlp_extractor_class, # NEW
                mlp_extractor_kwargs=self.mlp_extractor_kwargs, # NEW
                # use_sde=self.use_sde, # Not needed as it will be restored using dist_class/kwargs
                log_std_init=self.log_std_init,
                dist_class=self.dist_class, # NEW
                dist_kwargs=self.dist_kwargs, # NEW
                logit_norm_class=self.logit_norm_class, # NEW
                logit_norm_kwargs=self.logit_norm_kwargs, # NEW
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data
    
    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        # if 'device' in self.mlp_extractor_kwargs:
        #     msg = f"Device was passed to mlp_extractor_kwargs. Overwriting with class device value {self.device}"
        #     warnings.warn(msg)
        # self.mlp_extractor_kwargs['device'] = self.device

        # If a pre-computed feature extractor dim is not given or is None 
        # get get output dimensions from feature extractor.
        if self.mlp_extractor_kwargs.get('feature_dim') is None:
            self.mlp_extractor_kwargs['feature_dim'] = self.features_dim 
            
        self.mlp_extractor = self.mlp_extractor_class(**self.mlp_extractor_kwargs)
        
        # NOTE: Temp assert checks
        assert hasattr(self.mlp_extractor, 'policy_net')
        assert hasattr(self.mlp_extractor, 'value_net')
        assert hasattr(self.mlp_extractor, 'latent_dim_pi')
        assert hasattr(self.mlp_extractor, 'latent_dim_vf')
        assert hasattr(self.mlp_extractor, 'forward')
        assert hasattr(self.mlp_extractor, 'forward_actor')
        assert hasattr(self.mlp_extractor, 'forward_critic')

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # TODO: More dynamic way of computing this?
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        err_msg = "logit_norm_class can not be used with a continuous action distribution"
        if isinstance(self.action_dist, DiagGaussianDistribution):
            assert self.logit_norm_class is None, err_msg
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            assert self.logit_norm_class is None, err_msg
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
            if self.logit_norm_class is not None:
                self.action_net = nn.Sequential(self.action_net, self.logit_norm_class(**self.logit_norm_kwargs))
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the features extractor(s)
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor)
        else:
            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")