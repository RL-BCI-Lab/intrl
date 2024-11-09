import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Tuple,
    TypeVar,
    Union,
)
from copy import deepcopy
from functools import partial
from pdb import set_trace

import numpy as np
import torch as th
from gymnasium.spaces import Box
from stable_baselines3.common import policies, utils, vec_env
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution
)
from imitation.algorithms.bc import (
    RolloutStatsComputer,
    BatchIteratorWithEpochEndCallback,
    enumerate_batches
)
from imitation.algorithms.base import (
    AnyTransitions,    
    DemonstrationAlgorithm,
)
from imitation.util import util
from imitation.data import types
from imitation.util import logger as imit_logger

from intrl.common.sb3.policies import ActorCriticPolicy
from intrl.common.torch_layers import MetaDataModule
from intrl.algorithms.imitation.bc.log_manager import BCLogManager
from intrl.common.imitation.utils import make_data_loader_from_flattened
SelfBCEnsemble = TypeVar("SelfBCEnsemble", bound="BCEnsemble")
SelfBCEnsemblePolicy = TypeVar("SelfBCEnsemblePolicy", bound="BCEnsemblePolicy")
ActorCritics =  Union[ActorCriticPolicy, policies.ActorCriticPolicy]


@dataclasses.dataclass(frozen=True)
class BCMetrics:
    """Container for the different components of behavior cloning loss.
    
        selected_action_probs: Probabilities for selected actions
        
        action_logits: Logits for ALL actions.
        
        loss: NLL or cross-entropy loss value.
    """
    selected_action_probs: th.Tensor
    action_logits: th.Tensor
    loss: th.Tensor
    
    
# @dataclasses.dataclass(frozen=True)
# class BCLogitNormMetrics(BCMetrics):
#     """Container for the different components of behavior cloning loss."""
#     selected_action_probs_orig: th.Tensor 
#     logits_norm: th.Tensor


class BCLoss:
    def __call__(
        self,
        policy: ActorCritics,
        obs: th.Tensor,
        acts: th.Tensor,
    ) -> BCMetrics:
        obs = util.safe_to_tensor(obs)
        acts = util.safe_to_tensor(acts)
        
        # NOTE: Same as ActorCritic.get_distribution()
        action_logits = self.get_action_logits(policy=policy, obs=obs)
        action_dist = self.get_action_distribution(policy=policy, logits=action_logits)

        action_log_probs = action_dist.log_prob(acts)
        selected_action_probs = th.exp(action_log_probs)
        neg_log_prob = (-action_log_probs).mean()
        metrics = dict(
            selected_action_probs=selected_action_probs,
            action_logits=action_logits,
            loss=neg_log_prob,
        )
        
        return self.get_metrics(policy, metrics, "BCMetrics", BCMetrics)
    
    def get_metrics(self, policy, metrics, metric_name, base_class):
        metadata = {}
        if isinstance(policy.action_net, th.nn.Sequential):
            if isinstance(policy.action_net[-1], MetaDataModule):
                metadata = policy.action_net[-1].metadata

        self.metric_class =  dataclasses.make_dataclass(
            metric_name, 
            metadata.keys(), 
            bases=base_class if isinstance(base_class, (list, tuple)) else (base_class,),
            frozen=True
        )
     
        metrics = {**metrics, **metadata}
        return self.metric_class(**metrics)
    
    def get_action_distribution(self, policy, logits):
        if isinstance(policy.action_dist, (CategoricalDistribution, MultiCategoricalDistribution)):
            return policy.action_dist.proba_distribution(action_logits=logits) 
        elif isinstance(policy.action_dist, DiagGaussianDistribution):
            return policy.action_dist.proba_distribution(logits, policy.log_std)
        else:
            raise ValueError(f"Invalid action distribution {type(policy.action_dist)}")
    
    def get_action_logits(self, policy, obs):
        # Network(s) for extracting representation if needed
        features = policy.extract_features(obs)
        # Network(s) getting policy and value outputs
        if policy.share_features_extractor:
            latent_pi, _ = policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)
            # latent_vf = policy.mlp_extractor.forward_critic(vf_features)

        return policy.action_net(latent_pi)

# class BCLogitNormLoss(BCLoss):

#     def __init__(
#         self, 
#         norm_logits: Union[Callable, None] = lambda x: x,  
#     ):
#         super().__init__()
#         self.norm_logits = norm_logits
    
#     def __call__(
#         self,
#         policy: ActorCritics,
#         obs: th.Tensor,
#         acts: th.Tensor,
#     ) -> BCMetrics:
#         obs = util.safe_to_tensor(obs)
#         acts = util.safe_to_tensor(acts)
        
#         action_logits, action_logits_orig  = self.get_action_logits(policy=policy, obs=obs)
        
#         # DEBUG: Get original probabilities for comparison
#         # DO NOT use action_dist_orig as object reference will change once
#         # ts_action_logits is used!
#         action_probs_orig = None
#         if action_logits_orig is not None:
#             with th.no_grad():
#                 action_dist_orig = self.get_action_distribution(
#                     policy=policy,
#                     logits=action_logits_orig
#                 )
#                 action_probs_orig = action_dist_orig.distribution.probs
        
#         action_logits_norm = self.norm_logits(action_logits)
#         action_dist = self.get_action_distribution(policy=policy, logits=action_logits_norm)
#         log_prob = action_dist.log_prob(acts)
#         neglogp = -log_prob

#         metrics = dict(
#             action_probs_orig=action_probs_orig,
#             action_probs=action_dist.distribution.probs,
#             logits=action_logits,
#             logits_norm=action_logits_norm,
#             loss=neglogp.mean(),
#         )
        
#         return self.get_metrics(policy, metrics, "BCMetrics", BCMetrics)
    
#     def get_metrics(self, policy, metrics, metric_name, base_class):
#         metadata = {}
#         if isinstance(self.norm_logits, LogitNormalization):
#             metadata = self.norm_logits.metadata
#         elif isinstance(policy.action_net, th.nn.Sequential):
#             if isinstance(policy.action_net[-1], LogitNormalization):
#                 metadata = policy.action_net[-1].metadata

#         # Build dynamic data class based on metadata for norm_logits. If it norm_logits
#         # is not used use base class.
#         if self.metric_class is None:
#             self.metric_class =  dataclasses.make_dataclass(
#                 metric_name, 
#                 metadata.keys(), 
#                 bases=base_class if isinstance(base_class, (list, tuple)) else (base_class,),
#                 frozen=True
#             )
     
#         metrics = {**metrics, **metadata}
#         return self.metric_class(**metrics)
    
#     def get_action_distribution(self, policy, logits):
#         if isinstance(policy.action_dist, CategoricalDistribution):
#             return policy.action_dist.proba_distribution(action_logits=logits) 
#         else:
#             msg = "Invalid action distribution, 'CategoricalDistribution' is currently "\
#                   "the only supported action distribution."
#             raise ValueError(msg)
    
#     def get_action_logits(self, policy, obs):
#         # Network(s) for extracting representation if needed
#         features = policy.extract_features(obs)
#         # Network(s) getting policy and value outputs
#         if policy.share_features_extractor:
#             latent_pi, _ = policy.mlp_extractor(features)
#         else:
#             pi_features, _ = features
#             latent_pi = policy.mlp_extractor.forward_actor(pi_features)
#             # latent_vf = policy.mlp_extractor.forward_critic(vf_features)
            
#         # DEBUG: Used to get original probabilities if logit norm layers are added
#         orig = None
#         if isinstance(policy.action_net, th.nn.Sequential):
#             with th.no_grad():
#                 orig = policy.action_net[0](latent_pi)
        
#         return policy.action_net(latent_pi), orig


# TODO: Convert use policies.BasePolicy parent where user passes a list dicts containing
#       the name class reference and init kwargs. Saving and loading for automatic object
#       instantiation requires class reference to be passed not instance. See the following
#       https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L22
class BCEnsemblePolicy(th.nn.Module):
    """ True ensemble of policies where each policy has its own network and optimizer.
    
    """
    policy_types = (ActorCriticPolicy, policies.ActorCriticPolicy)
    
    def __init__(self, policy_specs: List[ActorCritics]):
        super().__init__()
        self.policy_specs = policy_specs
        self.n_policies = len(policy_specs)
        self.policies = self.build_policies()
        self.action_space = policy_specs[0]['kwargs']['action_space']
        self.observation_space = policy_specs[0]['kwargs']['observation_space']
        self._ensemble_compatibility_checks()

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0
    
    def build_policies(self):
        return th.nn.ModuleList(
            [s['class'](**s['kwargs']) for s in self.policy_specs]
        )
    
    def _ensemble_compatibility_checks(self):
        type_check = [isinstance(p, self.policy_types) for p in self.policies]
        assert np.all(type_check), "Ensembles must all be of type ActorCriticPolicy"
        
        obs_matching = [self.observation_space == p.observation_space for p in self.policies[1:]]
        assert np.all(obs_matching), "Ensemble observation spaces are not matching"
        
        act_matching = [self.action_space == p.action_space for p in self.policies[1:]]
        assert np.all(act_matching), "Ensemble action spaces are not matching"
    
    def get_action_dist(self):
        dist_class, dist_kwargs = self.policies[0].dist_class, self.policies[0].dist_kwargs
        return dist_class(**dist_kwargs)
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Use 1st policy parameters. Assumes all policies are the same class type using
        # the same hyper-parameters
        p = self.policies[0]
        # Switch to eval mode (this affects batch norm / dropout), akin to self.eval()
        [p.set_training_mode(False) for p in self.policies]
        # (Policy independent): This code simply uses observation space to modify
        # observations if needed.
        observation, vectorized_env = p.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        # (Policy independent): This code simply uses the unsacle_action() method
        # for sb3 policies which rescales actions based on env information only.
        if isinstance(self.action_space, Box):
            if p.squash_output:
                # Rescale to proper domain when using squashing
                actions = p.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
            
        # TODO: This needs to check if it was originally in training mode or not already
        [p.set_training_mode(True) for p in self.policies]
        
        return actions, state
    
    def _predict(self, observations, deterministic):
        if len(self.policies) == 1:
            return self.policies[0]._predict(observations, deterministic=deterministic)
        else:
            return self._ensemble_predict(observations, deterministic=deterministic)
        
    def _ensemble_predict(self, observations, deterministic):
        avg_probs = None
        dist = self.get_action_dist()
        for p in self.policies:
            action_dist = p.get_distribution(observations)
            probs = action_dist.distribution.probs
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs += probs
        avg_probs /= len(self.policies)
        
        return dist.proba_distribution(avg_probs).get_actions(deterministic=deterministic)
    
    def _get_policy_outputs(self, policy, obs):
        features = policy.extract_features(obs)
        if policy.share_features_extractor:
            latent_pi, latent_vf = policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = policy.mlp_extractor.forward_critic(vf_features)
            
        values = policy.value_net(latent_vf)
        
        action_dist = policy._get_action_dist_from_latent(latent_pi)
        probs = action_dist.distribution.probs
        
        entropy = action_dist.distribution.entropy()
        
        return values, probs, entropy
    
    def evaluate_actions(
        self, 
        obs: th.Tensor, 
        actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """       
        if len(self.policies) == 1:
            avg_values, avg_selected_log_prob, entropy = self.policies[0].evaluate_actions(obs, actions)
        else:
            for i, p in enumerate(self.policies):
                values, probs, entropy = self._get_policy_outputs(p, obs)
                if i == 0:
                    avg_values, avg_probs, avg_entropy = values, probs, entropy 
                else:
                    avg_probs += probs
                    avg_values += values
                    avg_entropy += entropy
            avg_values /= self.n_policies
            avg_probs /= self.n_policies
            avg_entropy /= self.n_policies
            avg_prob_dist = self.get_action_dist().proba_distribution(avg_probs)
            avg_selected_log_prob = avg_prob_dist.log_prob(actions)
            entropy = avg_prob_dist.entropy()

        return avg_values, avg_selected_log_prob, entropy
    
    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)
        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        policy_specs = deepcopy(self.policy_specs)
        # NOTE: Can not pickle local function which is what lr_schedule is supposed to be.
        for ps in policy_specs:
            if 'lr_schedule' in ps['kwargs']:
                ps['kwargs']['lr_schedule'] = self._dummy_schedule
                
        return dict(
            policy_specs=policy_specs
        )

    @classmethod
    def load(
        cls: Type[SelfBCEnsemblePolicy], 
        path: str, 
        device: Union[th.device, str] = "auto"
    ) -> SelfBCEnsemblePolicy:
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = utils.get_device(device)
        saved_variables = th.load(path, map_location=device)

        # Create policy object
        policy = cls(**saved_variables["data"])  
        # Load weights
        policy.load_state_dict(saved_variables["state_dict"])
        policy.to(device)
        return policy
     

class BCEnsemble(DemonstrationAlgorithm):
    def __init__(
        self,
        policy_specs: List[Tuple[policies.ActorCriticPolicy, Dict]],
        rng: np.random.Generator,
        *,
        loss_kwargs: Optional[Dict] = None,
        demonstrations: Optional[AnyTransitions] = None,
        split_demos: bool = True,
        batch_size: int = 128,
        minibatch_size: Optional[int] = None,
        device: Union[str, th.device] = "auto",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """
            policy_specs = [
                {
                    class: ActorCriticPolicy,
                    kwargs: dict(...)
                }
            ]
        """ 
        # TODO: Add policy spec check
        self._policy =  BCEnsemblePolicy(policy_specs)
        self._demo_data_loader: Optional[Iterable[types.TransitionMapping]] = None
        self.split_demos = split_demos
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("Batch size must be a divisible by minibatch size.")
        self.rng = rng
        self.device = utils.get_device(device)
        self.loss_kwargs = loss_kwargs or {}
        self.loss_calculator = self._build_loss(self.loss_kwargs)
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
        )
        self.log_manager = BCLogManager(self.logger)
        
    @property
    def policy(self) -> BCEnsemblePolicy:
        return self._policy
    
    def _build_loss(self, loss_kwargs):
        return BCLoss(**loss_kwargs)
    
    def set_demonstrations(self, demonstrations: AnyTransitions) -> None:
        self.logger.info(f"Number of state-action pairs: {len(demonstrations)}")
        data_loaders = []
        if self.split_demos and self.policy.n_policies != 1:
            transition_type = type(demonstrations)
            fields = demonstrations.__dict__.keys()
            indexes = np.arange(len(demonstrations))
            np.random.shuffle(indexes)
            splits = np.array_split(indexes, self.policy.n_policies)
  
            for s in splits:
                transition_kwargs = {f:getattr(demonstrations, f)[s] for f in fields}
                demo_split = transition_type(**transition_kwargs)
                if len(demo_split) < self.batch_size:
                    msg = f"batch_size {self.batch_size} is greater than demonstration split {len(demo_split)}"
                    self.logger.warn(msg)
                    
                data_loader = make_data_loader_from_flattened(
                    demo_split,
                    self.minibatch_size,
                    # If set to False, this will break batch and minibatch interaction when training
                    data_loader_kwargs=dict(drop_last=True)
                )
                data_loaders.append(data_loader) 
        else:  
            data_loaders.append(
                make_data_loader_from_flattened(
                    demonstrations,
                    self.minibatch_size,
                    # If set to False, this will break batch and minibatch interaction when training
                    # This is because a constant minibatch size is assumed and if data size 
                    # does not divide evenly by batch size then preprocessing will be skilled.
                    data_loader_kwargs=dict(drop_last=True)
                )
            )
        self._demo_data_loader = data_loaders
    # TODO: Rework training to be EITHER epoching or bactching?
    def train(
        self,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        log_interval: Optional[int] = None,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        reset_tensorboard: bool = False,
        policy_logger_prefix: Optional[str] = ''
    ) -> SelfBCEnsemble:
        assert self._demo_data_loader is not None, "No demonstrations were set."
        self.n_epochs = n_epochs
        self.mini_per_batch = self.batch_size // self.minibatch_size
        self.n_minibatches = n_batches * self.mini_per_batch if n_batches is not None else None
        self.on_epoch_end = on_epoch_end
     
        self._compute_rollout_stats = RolloutStatsComputer(
            log_rollouts_venv,
            log_rollouts_n_episodes,
        )
        self._train_ensemble(log_interval, reset_tensorboard, policy_logger_prefix)
        
        return self
        
    def _train_ensemble(
        self,
        log_interval: Optional[int] = None,
        reset_tensorboard: Optional [bool] = False,
        policy_logger_prefix: Optional[str] = ''
    ) -> None:
        assert len(self._demo_data_loader) == self._policy.n_policies or len(self._demo_data_loader) == 1
        for i, policy in enumerate(self._policy.policies):
            self.logger.info(f"Policy {i+1}")
            self.log_manager.name = f'{policy_logger_prefix}policy-{i+1}'
            if reset_tensorboard:
                self.log_manager.reset_tensorboard_steps()
            self.log_manager.log_epoch(0)
            policy.to(self.device)
            
            demo_loader = self._demo_data_loader[i if len(self._demo_data_loader) != 1 else 0]
            log_interval = log_interval or len(demo_loader.dataset) // self.batch_size

            demonstration_batches = BatchIteratorWithEpochEndCallback(
                demo_loader,
                self.n_epochs,
                self.n_minibatches,
                partial(self.call_on_epoch_end, on_epoch_end=self.on_epoch_end),
            )
            batches_with_stats = enumerate_batches(demonstration_batches)
            
            policy.optimizer.zero_grad()
            self._train_policy(
                policy=policy, 
                batches_with_stats=batches_with_stats,
                log_interval=log_interval,
            )
                
    def _train_policy(self, policy, batches_with_stats, log_interval):
        for (batch_num, mb_size, num_samples_so_far), batch in batches_with_stats:
            
            training_metrics = self._compute_loss(policy=policy, batch=batch)
            # NOTE: Renormalize the loss to be averaged over the whole
            #       batch size instead of the minibatch size.
            #       If there is an incomplete batch, its gradients will be
            #       smaller, which may be helpful for stability.
            loss = training_metrics.loss * mb_size / self.batch_size
            loss.backward()

            # Useful when batch_size needs to be split into smaller groups mb
            true_batch_num = batch_num * self.minibatch_size // self.batch_size
            
            # print(batch_num, true_batch_num, num_samples_so_far)
            if num_samples_so_far % self.batch_size == 0:
                self.process_batch(
                    policy=policy,
                    batch_num=true_batch_num,
                    log_interval=log_interval,
                    minibatch_size=mb_size,
                    num_samples_so_far=num_samples_so_far,
                    training_metrics=training_metrics,
                )
  
        # Run if there remains an incomplete batch
        if num_samples_so_far % self.batch_size != 0:
            batch_num += 1
            self.process_batch(
                    policy=policy,
                    batch_num=true_batch_num,
                    log_interval=log_interval,
                    minibatch_size=mb_size,
                    num_samples_so_far=num_samples_so_far,
                    training_metrics=training_metrics,
        
            )
    
    def _compute_loss(
        self, 
        policy: policies.ActorCriticPolicy, 
        batch: Dict
    ) -> dataclasses.dataclass:
        obs = th.as_tensor(batch["obs"], device=policy.device).detach()
        acts = th.as_tensor(batch["acts"], device=policy.device).detach()

        return self.loss_calculator(
            policy=policy,
            obs=obs,
            acts=acts,
        )
    
    def call_on_epoch_end(self, epoch_number: int, on_epoch_end=None, **kwargs):
        # Iterate epoch count in logger
        self.log_manager.log_epoch(epoch_number+1)
        
        if on_epoch_end is not None:
            on_epoch_end()
    
    def process_batch(
        self,
        policy,
        batch_num,
        log_interval,
        minibatch_size,
        num_samples_so_far,
        training_metrics,
    ):
        policy.optimizer.step()
        policy.optimizer.zero_grad()
        
        # NOTE: Initial policy params are not logged
        if batch_num != 0 and batch_num % log_interval == 0:
            # NOTE: Evaluates CURRENT policy
            rollout_stats = self._compute_rollout_stats(policy, self.rng)

            self.log_manager.log_batch(
                batch_num=batch_num,
                batch_size=minibatch_size,
                num_samples_so_far=num_samples_so_far,
                training_metrics=training_metrics,
                rollout_stats=rollout_stats,
                policy=policy
            )
            
    
    def save_policy(self, path: str) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            path: path to directory in which policy will be saved
        """
        self.policy.save(path)
        
    # def load_policy(self, path: List[str]) -> None:
    #     """Load policy from path.

    #     Args:
    #         path: path to directory in which policy will be saved
    #     """
    #     assert len(self.policy) == len(path)
        
    #     for policy, p in zip(self.policy, path):
    #         state_dict = th.load(p, utils.get_device('auto'))
    #         policy.load_state_dict(state_dict)