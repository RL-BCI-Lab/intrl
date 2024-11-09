"""
MIT License

Copyright (c) 2019-2022 Center for Human-Compatible AI and Google LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import dataclasses
from typing import (
    Dict,
    Optional,
    TypeVar,
    Union,
)
from pathlib import Path
from copy import deepcopy
from pdb import set_trace

import numpy as np
import torch as th
from stable_baselines3.common import policies
from imitation.util import util

from intrl.common.torch_layers import reset_parameters, compare_models
from intrl.algorithms.imitation.bc.bcensemble import (
    BCMetrics,
    BCLoss,
    ActorCritics,
    BCEnsemble,
    BCEnsemblePolicy
)


SelfBCNoise = TypeVar("SelfBCNoise", bound="BCNoise")


@dataclasses.dataclass(frozen=True)
class BCNoiseMetrics(BCMetrics):
    """Container for the different components of behavior cloning loss."""
    prior_selected_action_probs: th.Tensor


class BCNoiseLoss(BCLoss):
    """Computes the loss used for Behavioral Cloning from Noisy Demonstrations."""

    def __call__(
        self,
        policy: ActorCritics,
        obs: th.Tensor,
        acts: th.Tensor,
        prior_policy: ActorCritics = None,
    ) -> BCMetrics:
        """Calculate the supervised learning loss used to train the behavioral clone.
        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.
        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        obs = util.safe_to_tensor(obs)
        acts = util.safe_to_tensor(acts)
       
        action_logits = self.get_action_logits(policy=policy, obs=obs)
        action_dist = self.get_action_distribution(policy=policy, logits=action_logits)
        selected_action_probs = th.gather(
                action_dist.distribution.probs, 
                dim=1, 
                index=acts.reshape(-1, 1)
        ).squeeze()
        action_log_probs = action_dist.log_prob(acts)
        neg_log_prob = -action_log_probs

        # Compute prior policy probs
        with th.no_grad():
            if prior_policy is None:
                shape = selected_action_probs.shape
                prior_selected_action_probs = th.ones(shape).to(action_log_probs.get_device())
            else:
                _, prior_selected_log_prob, _ = prior_policy.evaluate_actions(obs, acts)
                prior_selected_action_probs = th.exp(prior_selected_log_prob)
            
        weighted_neg_log_prob = neg_log_prob * prior_selected_action_probs

        return BCNoiseMetrics(
            prior_selected_action_probs=prior_selected_action_probs,
            selected_action_probs=selected_action_probs,
            action_logits=action_logits,
            loss=weighted_neg_log_prob.mean(),
        )


class BCNoise(BCEnsemble):
    """Behavioral cloning from noisy demonstrations """

    def __init__(
        self, 
        copy_n_policies: int = None,  
        copy_random_policies: bool = True, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.copy_n_policies = copy_n_policies or len(self.policy.policies)
        assert self.copy_n_policies <= len(self.policy.policies)
        self.copy_random_policies = copy_random_policies
        self._prior_policy = None

    @property
    def prior_policy(self) -> policies.ActorCriticPolicy:
        return self._prior_policy
    
    def _build_loss(self, loss_kwargs):
        return BCNoiseLoss(**loss_kwargs)
    
    def train(self, iterations: Optional[int] = 5, **kwargs) -> SelfBCNoise:
        self.iterations = iterations
        return super(BCNoise, self).train(**kwargs)
        
    def _train_ensemble(
        self,
        log_interval: Optional[int] = None,
        reset_tensorboard: Optional [bool] = False,
        policy_logger_prefix: Optional[str] = ''
    ) -> None:
        # NOTE: Each iteration generates a brand new ensemble policy. No learning 
        #       continues from one iteration to the next. Only prior_policy contains
        #       information regarding previously learned policies.
        for m in range(self.iterations):
            self.logger.info(f"Iteration {m}")
            # Reinitialize policy parameters 
            if m > 0:
                # self._policy = BCEnsemblePolicy(policy_specs=self.policy.policy_specs)   
                # self.policy.policies = self._policy.build_policies()
                self.policy.apply(reset_parameters)
                self.policy.to(self.device)
                # TODO: Remove this compare for debugging
                for p in range(self.policy.n_policies):
                    compare_models(self.policy.policies[p], self.prior_policy.policies[p])
               
            # Run normal ensemble BC
            super(BCNoise, self)._train_ensemble(
                log_interval, 
                reset_tensorboard,
                policy_logger_prefix=f'{policy_logger_prefix}iter-{m}_'
            )
            # Copy ensemble policies
            self._prior_policy = self.copy_policy(               
                n=self.copy_n_policies,
                random=self.copy_random_policies,
                copy_to=self.prior_policy
            )
            self.prior_policy.to(self.device)
           
    def copy_policy(
        self, n: int = None, 
        random: bool = True, 
        copy_to: BCEnsemblePolicy = None
    ):
        """ Copies n policies either in order given or by random selection.
        
        """
      
        policy_idxs = np.arange(self.policy.n_policies)
        if n is None:
            n = len(policy_idxs)
            copy_idx = policy_idxs
        else:
            copy_idx = np.random.choice(policy_idxs, size=(n,), replace=False) if random else policy_idxs[:n]
            copy_idx.sort()
        
        if copy_to is None:
            copy_specs = [deepcopy(self.policy.policy_specs[i]) for i in copy_idx]
            policy_copy = BCEnsemblePolicy(policy_specs=copy_specs)
        else:
            assert isinstance(copy_to, BCEnsemblePolicy)
            assert copy_to.n_policies == n
            policy_copy = copy_to
            
        for pidx, cidx in enumerate(copy_idx):
            state_dict = deepcopy(self.policy.policies[cidx].state_dict())
            policy_copy.policies[pidx].load_state_dict(state_dict)
   
        return policy_copy
    
    def _compute_loss(
        self, 
        policy: policies.ActorCriticPolicy, 
        batch: Dict
    ) -> dataclasses.dataclass:
        obs = th.as_tensor(batch["obs"], device=policy.device).detach()
        acts = th.as_tensor(batch["acts"], device=policy.device).detach()

        return self.loss_calculator(
            policy=policy, 
            prior_policy=self.prior_policy, 
            obs=obs,
            acts=acts
        )
    
    def save_policy(self, path: str) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            path: path to directory in which policy will be saved
        """
        p_split = path.split('.')
        if len(p_split) == 1:
            p += ['{}', '.pt']
        else:
            p_split = p_split[:-1] + ['{}.', p_split[-1]]
        path = ''.join(p_split) 
        self.policy.save(path.format(''))
        self.prior_policy.save(path.format('-prior'))