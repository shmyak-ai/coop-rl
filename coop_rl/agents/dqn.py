# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact implementation of a DQN agent in JAx.

The differencies from vanilla dopamine buffer:
- delete absl logging since moving to ray distributed version

"""

import collections
import functools
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as onp
import optax
import ray
import tensorflow as tf

from coop_rl import losses, networks
from coop_rl.metrics import statistics_instance
from coop_rl.utils import (
    linearly_decaying_epsilon,
)


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11))
def train(
    network_def,
    online_params,
    target_params,
    optimizer,
    optimizer_state,
    states,
    actions,
    next_states,
    rewards,
    terminals,
    cumulative_gamma,
    loss_type="mse",
):
    """Run the training step."""

    def loss_fn(params, target):
        def q_online(state):
            return network_def.apply(params, state)

        q_values = jax.vmap(q_online)(states).q_values
        q_values = jnp.squeeze(q_values)
        replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
        if loss_type == "huber":
            return jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))
        return jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))

    def q_target(state):
        return network_def.apply(target_params, state)

    target = target_q(q_target, next_states, rewards, terminals, cumulative_gamma)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(online_params, target)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, params=online_params)
    online_params = optax.apply_updates(online_params, updates)
    return optimizer_state, online_params, loss


def target_q(target_network, next_states, rewards, terminals, cumulative_gamma):
    """Compute the target Q-value."""
    q_vals = jax.vmap(target_network, in_axes=(0))(next_states).q_values
    q_vals = jnp.squeeze(q_vals)
    replay_next_qt_max = jnp.max(q_vals, 1)
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    return jax.lax.stop_gradient(
        rewards + cumulative_gamma * replay_next_qt_max * (1.0 - terminals)
        )


@ray.remote(num_gpus=1, num_cpus=1)
class JaxDQNAgent:
    """A JAX implementation of the DQN agent."""

    def __init__(
        self,
        control_actor,
        replay_actor,
        optimizer,
        args_optimizer,
        num_actions,
        observation_shape,
        observation_dtype,
        stack_size,
        network=networks.NatureDQNNetwork,
        gamma=0.99,
        update_horizon=1,
        min_replay_history=20000,
        update_period=4,
        target_update_period=8000,
        epsilon_fn=linearly_decaying_epsilon,
        epsilon_train=0.01,
        epsilon_eval=0.001,
        epsilon_decay_period=250000,
        eval_mode=False,
        workdir=None,
        summary_writing_frequency=500,
        allow_partial_reload=False,
        seed=None,
        loss_type="mse",
        preprocess_fn=None,
        collector_allowlist=("tensorboard",),
    ):
        """Initializes the agent and constructs the necessary components.

        Note: We are using the Adam optimizer by default for JaxDQN, which differs
              from the original NatureDQN and the dopamine TensorFlow version. In
              the experiments we have ran, we have found that using Adam yields
              improved training performance.

        Args:
          num_actions: int, number of actions the agent can take at any state.
          observation_shape: tuple of ints describing the observation shape.
          observation_dtype: jnp.dtype, specifies the type of the observations.
          stack_size: int, number of frames to use in state stack.
          network: Jax network to use for training.
          gamma: float, discount factor with the usual RL meaning.
          update_horizon: int, horizon at which updates are performed, the 'n' in
            n-step update.
          min_replay_history: int, number of transitions that should be experienced
            before the agent begins training its value function.
          update_period: int, period between DQN updates.
          target_update_period: int, update period for the target network.
          epsilon_fn: function expecting 4 parameters: (decay_period, step,
            warmup_steps, epsilon). This function should return the epsilon value
            used for exploration during training.
          epsilon_train: float, the value to which the agent's epsilon is eventually
            decayed during training.
          epsilon_eval: float, epsilon used when evaluating the agent.
          epsilon_decay_period: int, length of the epsilon decay schedule.
          eval_mode: bool, True for evaluation and False for training.
          optimizer: str, name of optimizer to use.
          summary_writer: SummaryWriter object for outputting training statistics.
            May also be a str specifying the base directory, in which case the
            SummaryWriter will be created by the agent.
          summary_writing_frequency: int, frequency with which summaries will be
            written. Lower values will result in slower training.
          allow_partial_reload: bool, whether we allow reloading a partial agent
            (for instance, only the network parameters).
          seed: int, a seed for DQN's internal RNG, used for initialization and
            sampling actions. If None, will use the current time in nanoseconds.
          loss_type: str, whether to use Huber or MSE loss during training.
          preprocess_fn: function expecting the input state as parameter which it
            preprocesses (such as normalizing the pixel values between 0 and 1)
            before passing it to the Q-network. Defaults to None.
          collector_allowlist: list of str, if using CollectorDispatcher, this can
            be used to specify which Collectors to log to.
        """
        self.control_actor = control_actor
        self.replay_actor = replay_actor

        assert isinstance(observation_shape, tuple)
        seed = int(time.time() * 1e6) if seed is None else seed

        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size

        if preprocess_fn is None:
            self.network_def = network(num_actions=num_actions)
            self.preprocess_fn = networks.identity_preprocess_fn
        else:
            self.network_def = network(num_actions=num_actions, inputs_preprocessed=True)
            self.preprocess_fn = preprocess_fn

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.min_replay_history = min_replay_history
        self.target_update_period = target_update_period
        self.epsilon_fn = epsilon_fn
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.epsilon_decay_period = epsilon_decay_period
        self.update_period = update_period
        self.eval_mode = eval_mode
        self.training_steps = 0

        summary_writer_dir = os.path.join(workdir, "tensorboard/")
        self.summary_writer = tf.summary.create_file_writer(summary_writer_dir)
        self.summary_writing_frequency = summary_writing_frequency

        self.allow_partial_reload = allow_partial_reload
        self._loss_type = loss_type
        self._collector_allowlist = collector_allowlist

        self._rng = jax.random.key(seed)
        state_shape = self.observation_shape + (stack_size,)
        self.state = onp.zeros(state_shape)
        self._build_networks_and_optimizer(optimizer, args_optimizer)

        # Variables to be initialized by the agent once it interacts with the
        # environment.
        self._observation = None
        self._last_observation = None

    def _build_networks_and_optimizer(self, optimizer, args_optimizer):
        self._rng, rng = jax.random.split(self._rng)
        state = self.preprocess_fn(self.state)
        self.online_params = self.network_def.init(rng, x=state)
        self.optimizer = optimizer(**self._args_optimizer)
        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = self.online_params

    def _sample_from_replay_buffer(self):
        samples = self._replay.sample_transition_batch()
        types = self._replay.get_transition_elements()
        self.replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types, strict=False):
            self.replay_elements[element_type.name] = element

    def _sync_weights(self):
        """Syncs the target_network_params with online_params."""
        self.target_network_params = self.online_params

    def _train_step(self):
        """Runs a single training step.

        Runs training if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online_params to target_network_params if training
        steps is a multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                self._sample_from_replay_buffer()
                states = self.preprocess_fn(self.replay_elements["state"])
                next_states = self.preprocess_fn(self.replay_elements["next_state"])
                self.optimizer_state, self.online_params, loss = train(
                    self.network_def,
                    self.online_params,
                    self.target_network_params,
                    self.optimizer,
                    self.optimizer_state,
                    states,
                    self.replay_elements["action"],
                    next_states,
                    self.replay_elements["reward"],
                    self.replay_elements["terminal"],
                    self.cumulative_gamma,
                    self._loss_type,
                )
                if (
                    self.summary_writer is not None
                    and self.training_steps > 0
                    and self.training_steps % self.summary_writing_frequency == 0
                ):
                    with self.summary_writer.as_default():
                        tf.summary.scalar("HuberLoss", loss, step=self.training_steps)
                    self.summary_writer.flush()
                    if hasattr(self, "collector_dispatcher"):
                        self.collector_dispatcher.write(
                            [
                                statistics_instance.StatisticsInstance(
                                    "Loss", onp.asarray(loss), step=self.training_steps
                                ),
                            ],
                            collector_allowlist=self._collector_allowlist,
                        )
            if self.training_steps % self.target_update_period == 0:
                self._sync_weights()

        self.training_steps += 1
