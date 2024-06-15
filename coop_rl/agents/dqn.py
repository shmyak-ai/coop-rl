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
import logging
import math
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import ray
from flax import core, struct
from flax.metrics import tensorboard
from flax.training import train_state

from coop_rl import losses, networks


class TrainState(train_state.TrainState):
    key: jax.Array
    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    def update_target_params(self):
        return self.replace(target_params=self.params)


def create_train_state(rng, network, args_network, optimizer, args_optimizer, obs_shape):
    """Creates initial `TrainState`."""
    state_rng, init_rng = jax.random.split(rng)
    model = network(**args_network)
    params = model.init(init_rng, jnp.ones(obs_shape))["params"]
    tx = optimizer(**args_optimizer)
    return TrainState.create(apply_fn=model.apply, params=params, target_params=params, key=state_rng, tx=tx)


def target_q(state, observations, rewards, terminals, cumulative_gamma):
    """Compute the target Q-value."""
    q_vals = jnp.squeeze(state.apply_fn({"params": state.target_params}, x=observations).q_values)
    replay_next_qt_max = jnp.max(q_vals, 1)
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max * (1.0 - terminals))


@functools.partial(jax.jit, static_argnums=(6, 7))
def train(
    state,
    observations,
    next_observations,
    actions,
    rewards,
    terminals,
    cumulative_gamma,
    loss_type="mse",
):
    """Run the training step."""

    def loss_fn(params, target):
        q_values = jnp.squeeze(state.apply_fn({"params": params}, x=observations).q_values)
        replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
        if loss_type == "huber":
            return jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))
        return jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))

    target = target_q(state, next_observations, rewards, terminals, cumulative_gamma)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params, target)
    state = state.apply_gradients(grads=grad)
    return state, loss


class JaxDQNAgent:
    """A JAX implementation of the DQN agent."""

    def __init__(
        self,
        workdir,
        observation_shape,
        network,
        args_network,
        optimizer,
        args_optimizer,
        gamma,
        training_steps,
        batch_size,
        update_horizon,
        loss_type,
        target_update_period,
        synchronization_period,
        summary_writing_period,
        save_period,
        min_replay_history=20000,
        replay_actor=None,
        control_actor=None,
        handler_sampler=lambda *args, **kwargs: None,
        args_handler_sampler=None,
        seed=None,
        preprocess_fn=networks.identity_preprocess_fn,
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
        self.logger  = logging.getLogger("ray")

        self.workdir = workdir
        seed = int(time.time() * 1e6) if seed is None else seed

        if args_handler_sampler is None:
            args_handler_sampler = {"": None}
        self.control_actor = control_actor
        self.replay_actor = replay_actor
        self.sampler = handler_sampler(**args_handler_sampler)

        self.preprocess_fn = preprocess_fn
        self.min_replay_history = min_replay_history
        self.training_steps = training_steps

        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.batch_size = batch_size
        self._loss_type = loss_type

        self.target_update_period = target_update_period
        self.synchronization_period = synchronization_period
        self.summary_writing_period = summary_writing_period
        self.save_period = save_period

        self.summary_writer = tensorboard.SummaryWriter(os.path.join(workdir, "tensorboard/"))
        self.orbax_checkpointer = ocp.StandardCheckpointer()

        self.logger.info(f"Current devices: {jnp.arange(3).devices()}")
        # orbax so far cannot recognize a new key<fry> dtype, use the old one
        self._rng = jax.random.PRNGKey(seed)  # jax.random.key(seed)
        self._rng, rng = jax.random.split(self._rng)
        self.state = create_train_state(rng, network, args_network, optimizer, args_optimizer, observation_shape)

    def _train_step(self, replay_elements):
        """Runs a single training step.

        Runs training if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online_params to target_network_params if training
        steps is a multiple of target update period.
        """
        observations = self.preprocess_fn(replay_elements["state"])
        next_observations = self.preprocess_fn(replay_elements["next_state"])

        self.state, loss = train(
            self.state,
            observations,
            next_observations,
            replay_elements["action"],
            replay_elements["reward"],
            replay_elements["terminal"],
            self.cumulative_gamma,
            self._loss_type,
        )
        return loss

    def training_dopamine(self):
        transitions_processed = 0
        for training_step in itertools.count(start=1, step=1):
            replay_elements = self.replay_actor.sample_from_replay_buffer()
            self._train_step(replay_elements)
            transitions_processed += self.batch_size

            if training_step == self.training_steps:
                self.logger.info(f"Final training step {training_step} reached; finishing.")
                break

            if training_step % self.summary_writing_period == 0:
                self.logger.info(f"Training step: {training_step}.")
                self.logger.info(f"Transitions processed by the trainer: {transitions_processed}.")

            if training_step % self.target_update_period == 0:
                self.state = self.state.update_target_params()

    def training_reverb(self):
        transitions_processed = 0
        timer_fetching = []
        timer_sampling = []
        timer_training = []
        for training_step in itertools.count(start=1, step=1):
            start_timer = time.perf_counter()
            replay_elements, fetch_time = self.sampler.sample_from_replay_buffer()
            timer_sampling.append(time.perf_counter() - start_timer)
            timer_fetching.append(fetch_time)
            start_timer = time.perf_counter()
            loss = self._train_step(replay_elements)
            timer_training.append(time.perf_counter() - start_timer)
            transitions_processed += self.batch_size

            if training_step == self.training_steps:
                self.logger.info(f"Final training step {training_step} reached; finishing.")
                break

            if training_step % self.summary_writing_period == 0:
                self.logger.info(f"Training step: {training_step}.")
                self.logger.info(f"Transitions processed by the trainer: {transitions_processed}.")
                self.logger.debug(f"Fetching takes: {sum(timer_fetching) / len(timer_fetching):.4f}.")
                self.logger.info(f"Sampling takes: {sum(timer_sampling) / len(timer_sampling):.4f}.")
                self.logger.info(f"Training takes: {sum(timer_training) / len(timer_training):.4f}.")
                timer_fetching = []
                timer_sampling = []
                timer_training = []
                self.summary_writer.scalar("loss", loss, self.state.step)
                self.summary_writer.flush()

            if training_step % self.target_update_period == 0:
                self.state = self.state.update_target_params()

            if training_step % self.save_period == 0:
                self.orbax_checkpointer.save(os.path.join(self.workdir, f"chkpt_step_{training_step:07}"), self.state)

    def training_dopamine_remote(self):
        #  1. check if there are enough transitions in the replay buffer
        while True:
            add_count = ray.get(self.replay_actor.add_count.remote())
            self.logger.info(f"Add count: {add_count}.")
            if add_count >= self.min_replay_history:
                self.logger.info("Start training.")
                break
            else:
                self.logger.info("Waiting.")
                time.sleep(1)
        #  2. training
        transitions_processed = 0
        for training_step in itertools.count(start=1, step=1):
            replay_elements = ray.get(self.replay_actor.sample_from_replay_buffer.remote())
            self._train_step(replay_elements)
            transitions_processed += self.batch_size

            if training_step == self.training_steps:
                ray.get(self.control_actor.set_done.remote())
                self.logger.info(f"Final training step {training_step} reached; finishing.")
                break

            if training_step % self.summary_writing_period == 0:
                self.logger.info(f"Training step: {training_step}.")
                self.logger.info(f"Transitions processed by the trainer: {transitions_processed}.")
                self.logger.info(f"Transitions added to the buffer: {ray.get(self.replay_actor.add_count.remote())}.")

            if training_step % self.target_update_period == 0:
                self.state = self.state.update_target_params()

            if training_step % self.synchronization_period == 0:
                ray.get(self.control_actor.set_parameters.remote(self.online_params))

    def training_reverb_remote(self):
        #  1. check if there are enough transitions in the replay buffer
        while True:
            add_count = self.sampler.add_count()
            self.logger.info(f"Add count: {add_count}.")
            if add_count >= self.min_replay_history:
                self.logger.info("Start training.")
                break
            else:
                self.logger.info("Waiting.")
                time.sleep(1)
        #  2. training
        timer_fetching = []
        timer_sampling = []
        timer_training = []
        transitions_processed = 0
        for training_step in itertools.count(start=1, step=1):
            start_timer = time.perf_counter()
            replay_elements, fetch_time = self.sampler.sample_from_replay_buffer()
            timer_sampling.append(time.perf_counter() - start_timer)
            timer_fetching.append(fetch_time)
            start_timer = time.perf_counter()
            try:
                loss = self._train_step(replay_elements)
            except Exception as e:
                self.logger.debug(e)
            timer_training.append(time.perf_counter() - start_timer)
            transitions_processed += self.batch_size

            if training_step == self.training_steps:
                ray.get(self.control_actor.set_done.remote())
                self.logger.info(f"Final training step {training_step} reached; finishing.")
                break

            if training_step % self.summary_writing_period == 0:
                self.logger.info(f"Training step: {training_step}.")
                self.logger.info(f"Transitions processed by the trainer: {transitions_processed}.")
                self.logger.debug(f"Fetching takes: {sum(timer_fetching) / len(timer_fetching):.4f}.")
                self.logger.info(f"Sampling takes: {sum(timer_sampling) / len(timer_sampling):.4f}.")
                self.logger.info(f"Training takes: {sum(timer_training) / len(timer_training):.4f}.")
                timer_fetching = []
                timer_sampling = []
                timer_training = []
                self.summary_writer.scalar("loss", loss, self.state.step)
                self.summary_writer.flush()

            if training_step % self.target_update_period == 0:
                self.state = self.state.update_target_params()
                self.logger.info("Parameters sent.")

            if training_step % self.synchronization_period == 0:
                ray.get(self.control_actor.set_parameters.remote({"params": self.state.params}))

            if training_step % self.save_period == 0:
                self.orbax_checkpointer.save(os.path.join(self.workdir, f"chkpt_step_{training_step:07}"), self.state)
