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

import contextlib
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
    params = model.init(init_rng, jnp.ones((1, *obs_shape)))["params"]
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


class DQN:
    def __init__(
        self,
        *,
        trainer_seed,
        log_level="INFO",
        workdir,
        training_steps,
        loss_type,
        gamma,
        batch_size,
        update_horizon,
        target_update_period,
        summary_writing_period,
        save_period,
        synchronization_period,
        observation_shape,
        flax_state,
        buffer,
        controller,
        network,
        args_network,
        optimizer,
        args_optimizer,
        preprocess_fn=networks.identity_preprocess_fn,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.workdir = workdir

        self.controller = controller
        self.buffer = buffer

        self.preprocess_fn = preprocess_fn
        self.training_steps = training_steps

        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.batch_size = batch_size
        self.loss_type = loss_type

        self.target_update_period = target_update_period
        self.synchronization_period = synchronization_period
        self.summary_writing_period = summary_writing_period
        self.save_period = save_period

        self.summary_writer = tensorboard.SummaryWriter(os.path.join(workdir, "tensorboard/"))
        self.orbax_checkpointer = ocp.StandardCheckpointer()

        self.logger.info(f"Current devices: {jnp.arange(3).devices()}")
        self._rng = jax.random.PRNGKey(trainer_seed)
        if flax_state is None:
            self._rng, rng = jax.random.split(self._rng)
            self.flax_state = create_train_state(
                rng, network, args_network, optimizer, args_optimizer, observation_shape
            )

        self.futures = self.controller.set_parameters.remote(self.flax_state.params)

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

        self.flax_state, loss = train(
            self.flax_state,
            observations,
            next_observations,
            replay_elements["action"],
            replay_elements["reward"],
            replay_elements["terminal"],
            self.cumulative_gamma,
            self.loss_type,
        )

        with contextlib.suppress(TypeError):
            self.controller.set_parameters(self.flax_state.params)
        return loss

    def training(self):
        #  1. check if there are enough transitions in the replay buffer
        while True:
            try:
                add_count = self.sampler.add_count()
            except AttributeError:
                add_count = ray.get(self.buffer.add_count.remote())
            self.logger.info(f"Current buffer size: {add_count}.")
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
            try:
                replay_elements, fetch_time = self.sampler.sample_from_replay_buffer()
            except AttributeError:
                start = time.perf_counter()
                replay_elements = ray.get(self.buffer.sample_from_replay_buffer.remote())
                fetch_time = time.perf_counter() - start
            timer_sampling.append(time.perf_counter() - start_timer)
            timer_fetching.append(fetch_time)
            start_timer = time.perf_counter()
            # try:
            loss = self._train_step(replay_elements)
            # except Exception as e:
            #     self.logger.debug(e)
            timer_training.append(time.perf_counter() - start_timer)
            transitions_processed += self.batch_size

            if training_step == self.training_steps:
                ray.get(self.controller.set_done.remote())
                self.logger.info(f"Final training step {training_step} reached; finishing.")
                break

            if training_step % self.summary_writing_period == 0:
                store_size = ray.get(self.controller.store_size.remote())
                buffer_size = self.sampler.add_count()
                self.logger.info(f"Training step: {training_step}.")
                self.logger.debug(f"Weights store size: {store_size}.")
                self.logger.debug(f"Current buffer size: {buffer_size}.")
                self.logger.info(f"Transitions processed by the trainer: {transitions_processed}.")
                self.logger.debug(f"Fetching takes: {sum(timer_fetching) / len(timer_fetching):.4f}.")
                self.logger.debug(f"Sampling takes: {sum(timer_sampling) / len(timer_sampling):.4f}.")
                self.logger.debug(f"Training takes: {sum(timer_training) / len(timer_training):.4f}.")
                try:
                    add_count = self.sampler.add_count()
                except AttributeError:
                    add_count = ray.get(self.buffer.add_count.remote())
                self.logger.debug(f"Current buffer size: {add_count}.")
                timer_fetching = []
                timer_sampling = []
                timer_training = []
                self.summary_writer.scalar("loss", loss, self.flax_state.step)
                self.summary_writer.flush()

            if training_step % self.target_update_period == 0:
                self.flax_state = self.flax_state.update_target_params()

            if training_step % self.synchronization_period == 0:
                ray.get(self.futures)
                self.futures = self.controller.set_parameters.remote(self.flax_state.params)

            if training_step % self.save_period == 0:
                orbax_checkpoint_path = os.path.join(self.workdir, f"chkpt_train_step_{self.flax_state.step:07}")
                self.orbax_checkpointer.save(orbax_checkpoint_path, self.flax_state)
                self.logger.info(f"Orbax checkpoint is in: {orbax_checkpoint_path}")
                self.logger.info(f"Reverb checkpoint is in: {self.sampler.checkpoint()}")
