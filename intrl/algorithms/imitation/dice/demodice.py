import dataclasses
import pickle
import os
from functools import partial
from typing import (
    Any,
    Callable,
    Mapping,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Tuple,
    TypeVar,
    Union,
)
from pdb import set_trace

import numpy as np
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from stable_baselines3.common.vec_env import VecEnv 
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from gymnasium import spaces

from imitation.algorithms.base import (
    AnyTransitions,    
    DemonstrationAlgorithm,
)
from imitation.util import util
from imitation.data import types
from imitation.util import logger as imit_logger
from imitation.algorithms.bc import RolloutStatsComputer
    
from intrl.common.gym.utils import get_n_actions
from intrl.common.data.rollout import FakeRNG
from intrl.algorithms.imitation.dice import models 
from intrl.algorithms.imitation.dice.log_manager import DICELogManager


SelfDemoDICE = TypeVar("SelfDemoDICE", bound="DemoDICE")


@dataclasses.dataclass(frozen=True)
class DemoDICEMetrics:
    """Container for the different components of behavior cloning loss.

    """
    cost_loss: float
    nu_loss: float
    actor_loss: float
    expert_nu: float
    union_nu: float
    init_nu: float
    union_adv: float


def minimax_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    label_smoothing=0.25,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Original minimax discriminator loss for GANs, with label smoothing.

    Code Reference:
        https://github.com/tensorflow/gan

    Note that the authors don't recommend using this loss. A more practically
    useful loss is `modified_discriminator_loss`.

    L = - real_weights * log(sigmoid(D(x)))
        - generated_weights * log(1 - sigmoid(D(G(z))))

    See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
    details.

    Args:
        discriminator_real_outputs: Discriminator output on real data.
        discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
        label_smoothing: The amount of smoothing for positive labels. This technique
        is taken from `Improved Techniques for Training GANs`
        (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
        real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `real_data`, and must be broadcastable to `real_data` (i.e., all
        dimensions must be either `1`, or the same as the corresponding
        dimension).
        generated_weights: Same as `real_weights`, but for `generated_data`.
        scope: The scope for the operations performed in computing the loss.
        loss_collection: collection to which this loss will be added.
        reduction: A `tf.losses.Reduction` to apply to loss.
        add_summaries: Whether or not to add summaries for the loss.

    Returns:
        A loss Tensor. The shape depends on `reduction`.
  """
  with tf.compat.v1.name_scope(
      scope, 'discriminator_minimax_loss',
      (discriminator_real_outputs, discriminator_gen_outputs, real_weights,
       generated_weights, label_smoothing)) as scope:

    # -log((1 - label_smoothing) - sigmoid(D(x)))
    loss_on_real = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs),
        discriminator_real_outputs,
        real_weights,
        label_smoothing,
        scope,
        loss_collection=None,
        reduction=reduction)
    # -log(- sigmoid(D(G(x))))
    loss_on_generated = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_gen_outputs),
        discriminator_gen_outputs,
        generated_weights,
        scope=scope,
        loss_collection=None,
        reduction=reduction)

    loss = loss_on_real + loss_on_generated
    tf.compat.v1.losses.add_loss(loss, loss_collection)

    if add_summaries:
      tf.compat.v1.summary.scalar('discriminator_gen_minimax_loss',
                                  loss_on_generated)
      tf.compat.v1.summary.scalar('discriminator_real_minimax_loss',
                                  loss_on_real)
      tf.compat.v1.summary.scalar('discriminator_minimax_loss', loss)

  return loss


class DemoDICELoss:
    def __init__(
        self, 
        grad_reg_coeffs: Tuple[float] = (0.1, 1e-4), 
        alpha: float = 0.0, 
        discount: float = 0.99,
        eps: float = np.finfo(np.float32).eps, 
        eps2: float = 1e-3
    ):
        self.grad_reg_coeffs = grad_reg_coeffs
        self.non_expert_regularization = alpha + 1
        self.discount = discount
        self.eps = eps
        self.eps2 = eps2
        
    # @tf.function
    def __call__(
        self, 
        policy,
        init_obs, 
        expert_obs, 
        expert_acts, 
        expert_next_obs,
        union_obs, 
        union_acts, 
        union_next_obs
    ):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(policy.cost.variables)
            tape.watch(policy.actor.variables)
            tape.watch(policy.critic.variables)

            # define inputs
            expert_inputs = tf.concat([expert_obs, expert_acts], -1)
            union_inputs = tf.concat([union_obs, union_acts], -1)

            # call cost functions
            expert_cost_val, _ = policy.cost(expert_inputs)
            union_cost_val, _ = policy.cost(union_inputs)
            unif_rand = tf.random.uniform(shape=(expert_obs.shape[0], 1))
            mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * union_inputs
            mixed_inputs2 = unif_rand * tf.random.shuffle(union_inputs) + (1 - unif_rand) * union_inputs
            mixed_inputs = tf.concat([mixed_inputs1, mixed_inputs2], 0)

            # gradient penalty for cost
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(mixed_inputs)
                cost_output, _ = policy.cost(mixed_inputs)
                cost_output = tf.math.log(1 / (tf.nn.sigmoid(cost_output) + self.eps2) - 1 + self.eps2)
            cost_mixed_grad = tape2.gradient(cost_output, [mixed_inputs])[0] + self.eps
            cost_grad_penalty = tf.reduce_mean(
                tf.square(tf.norm(cost_mixed_grad, axis=-1, keepdims=True) - 1))
            cost_loss = minimax_discriminator_loss(expert_cost_val, union_cost_val, label_smoothing=0.) \
                        + self.grad_reg_coeffs[0] * cost_grad_penalty
            union_cost = tf.math.log(1 / (tf.nn.sigmoid(union_cost_val) + self.eps2) - 1 + self.eps2)

            # nu learning
            init_nu, _ = policy.critic(init_obs)
            expert_nu, _ = policy.critic(expert_obs)
            expert_next_nu, _ = policy.critic(expert_next_obs)
            union_nu, _ = policy.critic(union_obs)
            union_next_nu, _ = policy.critic(union_next_obs)
            union_adv_nu = - tf.stop_gradient(union_cost) + self.discount * union_next_nu - union_nu

            non_linear_loss = self.non_expert_regularization * tf.reduce_logsumexp(
                union_adv_nu / self.non_expert_regularization)
            linear_loss = (1 - self.discount) * tf.reduce_mean(init_nu)
            nu_loss = non_linear_loss + linear_loss

            # weighted BC
            weight = tf.expand_dims(tf.math.exp((union_adv_nu - tf.reduce_max(union_adv_nu)) / self.non_expert_regularization), 1)
            weight = weight / tf.reduce_mean(weight)
            pi_loss = - tf.reduce_mean(
                tf.stop_gradient(weight) * policy.actor.get_log_prob(union_obs, union_acts))

            # gradient penalty for nu
            if self.grad_reg_coeffs[1] is not None:
                unif_rand2 = tf.random.uniform(shape=(expert_obs.shape[0], 1))
                nu_inter = unif_rand2 * expert_obs + (1 - unif_rand2) * union_obs
                nu_next_inter = unif_rand2 * expert_next_obs + (1 - unif_rand2) * union_next_obs

                nu_inter = tf.concat([union_obs, nu_inter, nu_next_inter], 0)

                with tf.GradientTape(watch_accessed_variables=False) as tape3:
                    tape3.watch(nu_inter)
                    nu_output, _ = policy.critic(nu_inter)

                nu_mixed_grad = tape3.gradient(nu_output, [nu_inter])[0] + self.eps
                nu_grad_penalty = tf.reduce_mean(
                    tf.square(tf.norm(nu_mixed_grad, axis=-1, keepdims=True)))
                nu_loss += self.grad_reg_coeffs[1] * nu_grad_penalty

        nu_grads = tape.gradient(nu_loss, policy.critic.variables)
        cost_grads = tape.gradient(cost_loss, policy.cost.variables)
        pi_grads = tape.gradient(pi_loss, policy.actor.variables)

        policy.critic_optimizer.apply_gradients(zip(nu_grads, policy.critic.variables))
        policy.cost_optimizer.apply_gradients(zip(cost_grads, policy.cost.variables))
        policy.actor_optimizer.apply_gradients(zip(pi_grads, policy.actor.variables))
        del tape
        return dict(
            cost_loss=cost_loss,
            nu_loss=nu_loss,
            actor_loss=pi_loss,
            expert_nu=tf.reduce_mean(expert_nu),
            union_nu=tf.reduce_mean(union_nu),
            init_nu=tf.reduce_mean(init_nu),
            union_adv=tf.reduce_mean(union_adv_nu),
        )


class DemoDICEPolicy(tf.keras.layers.Layer):
    """ Class that implements DemoDICE training """
    def __init__(
        self,
        observation_space, 
        action_space, 
        hidden_size: int = 256,
        critic_lr: float = 3e-4,
        actor_lr: float =  3e-4, 
        use_last_layer_bias_cost: bool = False,
        kernel_initializer: str = 'he_normal',
    ):
        super(DemoDICEPolicy, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_dim = get_action_dim(action_space)
        self.n_actions = get_n_actions(action_space)
        state_dims = get_obs_shape(observation_space)
        if len(state_dims) > 1:
            msg = f'Detected state dimensions of {state_dims} but only 1D state dimensions are currently supported.'
            raise ValueError(msg)
        self.state_dim = state_dims[0]
        
        self.hidden_size = hidden_size
        self.use_last_layer_bias_cost = use_last_layer_bias_cost
        self.kernel_initializer = kernel_initializer

        self.cost = models.Critic(
            self.state_dim, 
            self.action_dim, 
            hidden_size=hidden_size,
            use_last_layer_bias=use_last_layer_bias_cost,
            kernel_initializer=kernel_initializer
        )
        self.critic = models.Critic(
            self.state_dim, 
            0,
            hidden_size=hidden_size,
            use_last_layer_bias=use_last_layer_bias_cost,
            kernel_initializer=kernel_initializer
        )
        if isinstance(action_space, spaces.Discrete):
            self.actor = models.DiscreteActor(self.state_dim, self.n_actions, hidden_size=hidden_size)
        elif isinstance(action_space, spaces.Box):
            self.actor = models.TanhActor(self.state_dim, self.action_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Action shape {type(action_space)} not supported")

        self.cost.create_variables()
        self.critic.create_variables()
        self.actor.create_variables()

        self.cost_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        
    # @tf.function
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """ Prediction method with Ducktyping for SB3 
        
            This method is typically used when running evaluation rollouts
        """
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        all_actions, _ = self.actor(observation)
        if deterministic:
            actions = all_actions[0] # Greedy
        else:
            actions = all_actions[1] # Random based on logit dist

        return actions.numpy().reshape((-1, *self.action_space.shape)).astype(int), state

    def get_training_state(self):
        return {
            'cost_params': [(variable.name, variable.value().numpy()) for variable in self.cost.variables],
            'critic_params': [(variable.name, variable.value().numpy()) for variable in self.critic.variables],
            'actor_params': [(variable.name, variable.value().numpy()) for variable in self.actor.variables],
            'cost_optimizer_state': [(variable.name, variable.value().numpy()) for variable in self.cost_optimizer.variables()],
            'critic_optimizer_state': [(variable.name, variable.value().numpy()) for variable in self.critic_optimizer.variables()],
            'actor_optimizer_state': [(variable.name, variable.value().numpy()) for variable in self.actor_optimizer.variables()],
        }

    def save(self, path):
        print('Save checkpoint: ', path)
        data = {
            'training_state': self.get_training_state(),
            **self._get_constructor_parameters()
        }
        with open(path + '.tmp', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(path + '.tmp', path)
        print('Saved!')

    def _get_constructor_parameters(self)  -> Dict[str, Any]:
         return dict(
            observation_space=self.observation_space, 
            action_space=self.action_space,
            hidden_size=self.hidden_size,
            critic_lr=self.critic_optimizer.learning_rate,
            actor_lr=self.actor_optimizer.learning_rate,
            use_last_layer_bias_cost=self.use_last_layer_bias_cost,
            kernel_initializer=self.kernel_initializer
        )
         
    # def init_dummy(self):
    #     # dummy train_step (to create optimizer variables)
    #     dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
    #     dummy_action = np.zeros((1, self.action_dim), dtype=np.float32)
    #     dummy_union = tf.concat([dummy_state, dummy_action], -1)

    #     # self.update(dummy_state, dummy_state, dummy_action, dummy_state, dummy_state, dummy_action, dummy_state)
    #     self.critic(dummy_union)
    #     self.cost(dummy_union)
    #     self.actor(dummy_state)
        
    @classmethod
    def load(cls, path: str):
        print('Load checkpoint:', path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        training_state = data.pop('training_state')
        policy = cls(**data)
        # policy.init_dummy()
        policy.set_training_state(training_state)
        return policy

    def set_training_state(self, training_state):
        def _assign_values(variables, params):
            if len(variables) != len(params):
                raise ValueError
            assert len(variables) == len(params)
            for variable, (name, value) in zip(variables, params):
                assert variable.name == name
                variable.assign(value)

        _assign_values(self.cost.variables, training_state['cost_params'])
        _assign_values(self.critic.variables, training_state['critic_params'])
        _assign_values(self.actor.variables, training_state['actor_params'])
        # TODO: Can not initialize optimizers yet, must create variables somehow.
        # _assign_values(self.cost_optimizer.variables(), training_state['cost_optimizer_state'])
        # _assign_values(self.critic_optimizer.variables(), training_state['critic_optimizer_state'])
        # _assign_values(self.actor_optimizer.variables(), training_state['actor_optimizer_state'])
    
        
class DemoDICE(DemonstrationAlgorithm):
    def __init__(
        self,
        observation_space, 
        action_space, 
        demonstrations: Optional[Mapping[str, AnyTransitions]] = None,
        loss_kwargs: Optional[Dict] = None,
        batch_size: int = 128,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        self._policy =  DemoDICEPolicy(
            observation_space=observation_space, 
            action_space=action_space, 
        )
        self.batch_size = batch_size
        self.loss_kwargs = loss_kwargs or {}
        self.loss_calculator = self._build_loss(self.loss_kwargs)
        super().__init__(
                demonstrations=demonstrations,
                custom_logger=custom_logger,
            )
        self.log_manager = DICELogManager(
            self.logger,
            name='policy',
            networks_to_track=['cost', 'critic', 'actor']
        )
        
    @property
    def policy(self) -> DemoDICEPolicy:
        return self._policy
    
    def _build_loss(self, loss_kwargs):
        return DemoDICELoss(**loss_kwargs)

    def _concat_transitions_field(self, field):
        return np.concatenate([
            getattr(self._imperfect_transitions, field), 
            getattr(self._expert_transitions, field), 
        ]).astype(np.float32)
    
    def _get_init_obs(self, transitions):
        """ Get initial observation using dones
        
            Note:
                Assumes dones = terminal + truncated. If both terminal and truncated
                are not contained within dones, then captured initial observations will 
                be incomplete.
        """
        init_obs = [transitions.obs[0]]
        done_locs = np.where(transitions.dones == True)[0]
        for loc in done_locs:
            if loc+1 >= len(transitions):
                continue
            init_obs.append(transitions.obs[loc+1])
        # done = terminal + truncated = number of episodes = number of init obs
        assert len(init_obs) == transitions.dones.sum()
        
        return np.vstack(init_obs)
    
    def set_demonstrations(self, demonstrations: Dict[str, AnyTransitions]) -> None:
        assert isinstance(demonstrations, dict)
        assert 'expert' in demonstrations
        assert 'imperfect' in demonstrations
        
        self._expert_transitions = demonstrations['expert']
        self._imperfect_transitions = demonstrations['imperfect']
        expert_init_obs = self._get_init_obs(self._expert_transitions)
        imperfect_init_obs = self._get_init_obs(self._imperfect_transitions)
        
        self.union_init_obs = np.concatenate([imperfect_init_obs, expert_init_obs])
        self.union_obs = self._concat_transitions_field('obs')
        self.union_acts = self._concat_transitions_field('acts')
        self.union_next_obs = self._concat_transitions_field('next_obs')
        self.union_dones = self._concat_transitions_field('dones')

        self.logger.info(f'Total expert samples: {len(self._expert_transitions)}')
        self.logger.info(f'Total imperfect samples: {len(self._imperfect_transitions)}')
        if self.batch_size > len(self.union_init_obs):
            msg = f"batch_size {self.batch_size} is greater than number of initial obs {len(self.union_init_obs)}"
            self.logger.warn(msg)
    
    def train(
        self,
        n_batches: Optional[int] = 1000, # Number of training batches/iterations
        log_interval: Optional[int] = 100,
        log_rollouts_venv: Optional[VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        reset_tensorboard: bool = False,
    ) -> SelfDemoDICE:
        if reset_tensorboard:
            self.log_manager.reset_tensorboard_steps()
        self.log_manager.log_epoch(0)

        assert isinstance(log_rollouts_venv, VecEnv)
        self._compute_rollout_stats = RolloutStatsComputer(
            log_rollouts_venv,
            log_rollouts_n_episodes,
        )

        for batch_num in tqdm(range(n_batches)):
            batch = self.get_batch(self.batch_size)
            training_metrics = self._compute_loss(batch)

            if batch_num != 0 and batch_num % log_interval == 0:
                # NOTE: Evaluates CURRENT policy
                rollout_stats = self._compute_rollout_stats(
                    partial(self.policy.predict, deterministic=True), 
                    FakeRNG # Do not need to shuffle trajectories
                )

                self.log_manager.log_batch(
                    batch_num=batch_num,
                    num_samples_so_far=self.batch_size*(batch_num+1),
                    training_metrics=training_metrics,
                    rollout_stats=rollout_stats,
                    policy=self.policy
                )
        return self
    
    def get_batch(self, batch_size):
        init_batch_size = batch_size
        if batch_size > len(self.union_init_obs):
            init_batch_size = len(self.union_init_obs)
        union_init_idx = np.random.randint(0, len(self.union_init_obs), size=init_batch_size)
        union_idx = np.random.randint(0, len(self.union_obs), size=batch_size)
        expert_idx = np.random.randint(0, len(self._expert_transitions), size=batch_size)

        return dict(
            init_obs=self.union_init_obs[union_init_idx].astype(np.float32),
            expert_obs=self._expert_transitions.obs[expert_idx].astype(np.float32), 
            expert_acts=self._expert_transitions.acts[expert_idx].astype(np.float32).reshape(-1, 1), 
            expert_next_obs=self._expert_transitions.next_obs[expert_idx].astype(np.float32),
            union_obs=self.union_obs[union_idx].astype(np.float32), 
            union_acts=self.union_acts[union_idx].astype(np.float32).reshape(-1, 1),
            union_next_obs=self.union_next_obs[union_idx].astype(np.float32), 
        )
        
    def _compute_loss(self, batch) -> dataclasses.dataclass:
        # tf.function does not allow dataclass to be returned so wrap it here
        return DemoDICEMetrics(**self.loss_calculator(policy=self.policy, **batch))
    
            
    def save_policy(self, path: str) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            path: path to directory in which policy will be saved
        """
        self.policy.save(path)