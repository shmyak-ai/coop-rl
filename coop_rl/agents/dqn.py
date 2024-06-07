# Copyright 2024 The Coop RL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact implementation of a DQN agent in JAx.

Modifications to the vanilla:
- keep only training related functionality
- make the agent a ray actor
- refactor

"""

import functools
import itertools
import math
import time

import jax
import jax.numpy as jnp
import numpy as onp
import optax
import ray

# import os
# import tensorflow as tf
from coop_rl import losses, networks


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
    return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max * (1.0 - terminals))


class JaxDQNAgent:
    """A JAX implementation of the DQN agent."""

    def __init__(
        self,
        workdir,
        optimizer,
        args_optimizer,
        training_steps,
        num_actions,
        observation_shape,
        network,
        gamma,
        batch_size,
        update_horizon,
        loss_type,
        target_update_period,
        synchronization_period,
        summary_writing_period,
        min_replay_history=20000,
        replay_actor=None,
        control_actor=None,
        seed=None,
        preprocess_fn=None,
    ):
        """Initializes the agent and constructs the necessary components.

        Note: We are using the Adam optimizer by default for JaxDQN, which differs
              from the original NatureDQN and the dopamine TensorFlow version. In
              the experiments we have ran, we have found that using Adam yields
              improved training performance.

        Args:
          optimizer: str, name of optimizer to use.
          num_actions: int, number of actions the agent can take at any state.
          observation_shape: tuple of ints describing the observation shape.
          network: Jax network to use for training.
          gamma: float, discount factor with the usual RL meaning.
          update_horizon: int, horizon at which updates are performed, the 'n' in
            n-step update.
          loss_type: str, whether to use Huber or MSE loss during training.
          target_update_period: int, update period for the target network.
          summary_writing_frequency: int, frequency with which summaries will be
            written. Lower values will result in slower training.
          seed: int, a seed for DQN's internal RNG, used for initialization and
            sampling actions. If None, will use the current time in nanoseconds.
          preprocess_fn: function expecting the input state as parameter which it
            preprocesses (such as normalizing the pixel values between 0 and 1)
            before passing it to the Q-network. Defaults to None.
        """
        self.control_actor = control_actor
        self.replay_actor = replay_actor

        self.min_replay_history = min_replay_history
        self.training_steps = training_steps

        assert isinstance(observation_shape, tuple)
        seed = int(time.time() * 1e6) if seed is None else seed

        if preprocess_fn is None:
            self.network = network(num_actions=num_actions)
            self.preprocess_fn = networks.identity_preprocess_fn
        else:
            self.network = network(num_actions=num_actions, inputs_preprocessed=True)
            self.preprocess_fn = preprocess_fn

        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.batch_size = batch_size

        # summary_writer_dir = os.path.join(workdir, "tensorboard/")
        # self.summary_writer = tf.summary.create_file_writer(summary_writer_dir)
        self.summary_writing_period = summary_writing_period

        self._loss_type = loss_type
        self.target_update_period = target_update_period
        self.synchronization_period = synchronization_period

        print(f"Current devices: {jnp.arange(3).devices()}")
        self._rng = jax.random.key(seed)
        self._build_networks_and_optimizer(observation_shape, optimizer, args_optimizer)

    def _build_networks_and_optimizer(self, observation_shape, optimizer, args_optimizer):
        self._rng, rng = jax.random.split(self._rng)
        state = self.preprocess_fn(onp.zeros(observation_shape))
        self.online_params = self.network.init(rng, x=state)
        self.optimizer = optimizer(**args_optimizer)
        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = self.online_params

    def _train_step(self, replay_elements):
        """Runs a single training step.

        Runs training if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online_params to target_network_params if training
        steps is a multiple of target update period.
        """
        states = self.preprocess_fn(replay_elements["state"])
        next_states = self.preprocess_fn(replay_elements["next_state"])

        self.optimizer_state, self.online_params, loss = train(
            self.network,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            states,
            replay_elements["action"],
            next_states,
            replay_elements["reward"],
            replay_elements["terminal"],
            self.cumulative_gamma,
            self._loss_type,
        )

    def training_dopamine(self):
        transitions_processed = 0
        for training_step in itertools.count(start=1, step=1):
            replay_elements = self.replay_actor.sample_from_replay_buffer()
            self._train_step(replay_elements)
            transitions_processed += self.batch_size

            if training_step == self.training_steps:
                print(f"Final training step {training_step} reached; finishing.")
                break

            if training_step % self.summary_writing_period == 0:
                print(f"Training step: {training_step}.")
                print(f"Transitions processed by the trainer: {transitions_processed}.")

            if training_step % self.target_update_period == 0:
                self.target_network_params = self.online_params

    def training_dopamine_remote(self):
        #  1. check if there are enough transitions in the replay buffer
        while True:
            add_count = ray.get(self.replay_actor.add_count.remote())
            print(f"Add count: {add_count}.")
            if add_count >= self.min_replay_history:
                print("Start training.")
                break
            else:
                print("Waiting.")
                time.sleep(1)
        #  2. training
        transitions_processed = 0
        for training_step in itertools.count(start=1, step=1):
            replay_elements = ray.get(self.replay_actor.sample_from_replay_buffer.remote())
            self._train_step(replay_elements)
            transitions_processed += self.batch_size

            if training_step == self.training_steps:
                ray.get(self.control_actor.set_done.remote())
                print(f"Final training step {training_step} reached; finishing.")
                break

            if training_step % self.summary_writing_period == 0:
                print(f"Training step: {training_step}.")
                print(f"Transitions processed by the trainer: {transitions_processed}.")
                print(f"Transitions added to the buffer: {ray.get(self.replay_actor.add_count.remote())}.")

            if training_step % self.target_update_period == 0:
                self.target_network_params = self.online_params

            if training_step % self.synchronization_period == 0:
                ray.get(self.control_actor.set_parameters.remote(self.online_params))
