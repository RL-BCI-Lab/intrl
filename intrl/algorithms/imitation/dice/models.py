from pdb import set_trace

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network, utils
from tf_agents.specs.tensor_spec import TensorSpec

class TanhActor(network.Network):
    def __init__(self, state_dim, action_dim, hidden_size=256, name='TanhNormalPolicy',
                 mean_range=(-7., 7.), logstd_range=(-5., 2.), eps=np.finfo(np.float32).eps, initial_std_scaler=1,
                 kernel_initializer='he_normal', activation_fn=tf.nn.relu):
        self._input_specs = TensorSpec(state_dim)
        self._action_dim = action_dim
        self._initial_std_scaler = initial_std_scaler

        super(TanhActor, self).__init__(self._input_specs, state_spec=(), name=name)

        hidden_sizes = (hidden_size, hidden_size)

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=activation_fn,
                                           kernel_initializer=kernel_initializer, name='mlp')
        self._fc_mean = tf.keras.layers.Dense(action_dim, name='policy_mean/dense',
                                              kernel_initializer=kernel_initializer)
        self._fc_logstd = tf.keras.layers.Dense(action_dim, name='policy_logstd/dense',
                                                kernel_initializer=kernel_initializer)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def call(self, inputs, step_type=(), network_state=(), training=True):
        del step_type  # unused
        h = inputs
        for layer in self._fc_layers:
            h = layer(h, training=training)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        logstd = self._fc_logstd(h)
        logstd = tf.clip_by_value(logstd, self.logstd_min, self.logstd_max)
        std = tf.exp(logstd) * self._initial_std_scaler
        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        pretanh_action = pretanh_action_dist.sample()
        action = tf.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return (tf.tanh(mean), action, log_prob), network_state

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = tf.tanh(pretanh_action)
        else:
            pretanh_action = tf.atanh(tf.clip_by_value(action, -1 + self.eps, 1 - self.eps))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - tf.reduce_sum(tf.math.log(1 - action ** 2 + self.eps), axis=-1)

        return log_prob, pretanh_log_prob

    def get_log_prob(self, states, actions):
        """Evaluate log probs for actions conditined on states.
        Args:
            states: A batch of states.
            actions: A batch of actions to evaluate log probs on.
        Returns:
            Log probabilities of actions.
        """
        h = states
        for layer in self._fc_layers:
            h = layer(h, training=True)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        logstd = self._fc_logstd(h)
        logstd = tf.clip_by_value(logstd, self.logstd_min, self.logstd_max)
        std = tf.exp(logstd) * self._initial_std_scaler

        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        pretanh_actions = tf.atanh(tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps))
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_actions)

        log_probs = pretanh_log_prob - tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.eps), axis=-1)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return log_probs


class DiscreteActor(network.Network):
    def __init__(self, state_dim, action_dim, hidden_size=256, name='DiscretePolicy',
                 kernel_initializer='he_normal', activation_fn=tf.nn.relu):
        self._input_specs = TensorSpec(state_dim)
        self._action_dim = action_dim

        super(DiscreteActor, self).__init__(self._input_specs, state_spec=(), name=name)

        hidden_sizes = (hidden_size, hidden_size)

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=activation_fn, kernel_initializer=kernel_initializer, name='mlp')
        self._logit_layer = tf.keras.layers.Dense(action_dim, name='logits/dense', kernel_initializer=kernel_initializer)

    def call(self, inputs, step_type=(), network_state=(), training=True):
        h = inputs
        for layer in self._fc_layers:
            h = layer(h, training=training)
        logits = self._logit_layer(h)
        dist = tfp.distributions.OneHotCategorical(logits)
        
        onehot_action = tf.cast(dist.sample(), tf.float32)
        action = tf.argmax(onehot_action, axis=1)
        log_prob = dist.log_prob(onehot_action)
        greedy_action = tf.argmax(logits, axis=1)
        
        return (greedy_action, action, log_prob), network_state

    def get_log_prob(self, states, actions, training=True):
        """Evaluate log probs for actions conditined on states.
        Args:
          states: A batch of states.
          actions: A batch of actions to evaluate log probs on.
        Returns:
          Log probabilities of actions.
        """
        h = states
        for layer in self._fc_layers:
            h = layer(h, training=training)

        logits = self._logit_layer(h)
        dist = tfp.distributions.Categorical(logits=logits)
        log_probs = tf.expand_dims(dist.log_prob(actions.flatten()), -1)  # To avoid broadcasting?

        return log_probs


class Critic(network.Network):
    def __init__(self, state_dim, action_dim, hidden_size=256, output_activation_fn=None, use_last_layer_bias=False,
                 output_dim=None, kernel_initializer='he_normal', name='ValueNetwork'):
        self._input_specs = TensorSpec(state_dim + action_dim)
        self._output_dim = output_dim

        super(Critic, self).__init__(self._input_specs, state_spec=(), name=name)

        hidden_sizes = (hidden_size, hidden_size)

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=tf.nn.relu,
                                           kernel_initializer=kernel_initializer, name='mlp')
        if use_last_layer_bias:
            last_layer_initializer = tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
            self._last_layer = tf.keras.layers.Dense(output_dim or 1, activation=output_activation_fn,
                                                     kernel_initializer=last_layer_initializer,
                                                     bias_initializer=last_layer_initializer, name='value')
        else:
            self._last_layer = tf.keras.layers.Dense(output_dim or 1, activation=output_activation_fn, use_bias=False,
                                                     kernel_initializer=kernel_initializer, name='value')

    def call(self, inputs, step_type=(), network_state=(), training=False):
        del step_type  # unused
        h = inputs
        for layer in self._fc_layers:
            h = layer(h, training=training)
        h = self._last_layer(h)

        if self._output_dim is None:
            h = tf.reshape(h, [-1])

        return h, network_state

