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

import functools
import itertools
import logging
import math
import os
import threading
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import ray
from flax import core, struct
from flax.metrics import tensorboard
from flax.training import train_state

from coop_rl import losses


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11))
def select_action(
    network_def,
    params,
    state,
    rng,
    num_actions,
    eval_mode,
    epsilon_eval,
    epsilon_train,
    epsilon_decay_period,
    step,
    warmup_steps,
    epsilon_fn,
):
    epsilon = jnp.where(
        eval_mode,
        epsilon_eval,
        epsilon_fn(
            epsilon_decay_period,
            step,
            warmup_steps,
            epsilon_train,
        ),
    )

    state = jnp.expand_dims(state, axis=0)
    rng, rng1, rng2 = jax.random.split(rng, num=3)
    p = jax.random.uniform(rng1)
    return (
        rng,
        jnp.where(
            p <= epsilon,
            jax.random.randint(rng2, (), 0, num_actions),
            jnp.argmax(network_def.apply(params, state).q_values),
        ),
        epsilon,
    )


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


def restore_dqn_flax_state(num_actions, network, optimizer, observation_shape, learning_rate, eps, checkpointdir):
    orbax_checkpointer = ocp.StandardCheckpointer()
    args_network = {"num_actions": num_actions}
    args_optimizer = {"learning_rate": learning_rate, "eps": eps}
    rng = jax.random.PRNGKey(0)  # jax.random.key(0)
    state = create_train_state(rng, network, args_network, optimizer, args_optimizer, observation_shape)
    abstract_my_tree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    return orbax_checkpointer.restore(checkpointdir, args=ocp.args.StandardRestore(abstract_my_tree))


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
        args_buffer,
        network,
        args_network,
        optimizer,
        args_optimizer,
        controller,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.workdir = workdir

        self.buffer = buffer(**args_buffer)

        self.training_steps = training_steps

        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self._cumulative_discount_vector = jnp.array(
            [math.pow(gamma, n) for n in range(update_horizon + 1)],
            dtype=jnp.float32,
        )
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
        else:
            self.flax_state = flax_state

        self.controller = controller
        self.futures = self.controller.set_parameters.remote(self.flax_state.params)
        self.buffer_lock = threading.Lock()

        self.traj_store = {}
        self.add_batch_size = args_buffer.add_batch_size
        self.store_lock = threading.Lock()

        self.is_done = False

    def add_traj_seq(self, data):
        # a trick to start a ray thread, is it necessary?
        if data == 1:
            return
        # save data from a collector
        collector_seed, *_data = data
        with self.store_lock:
            # for collectors synchronization, put only if there is no data already
            if self.traj_store.get(collector_seed) is not None:
                return False
            self.traj_store[collector_seed] = _data
            return True

    def buffer_updating(self):
        while True:
            if self.is_done:
                self.logger.info("Done signal received; finishing buffer updating.")
                break

            with self.store_lock:
                trajectories = [self.traj_store[key] for key in self.traj_store if self.traj_store[key]]

            if len(trajectories) != self.add_batch_size:
                time.sleep(0.1)
                continue

            with self.store_lock:
                self.traj_store = {}

            transposed = list(zip(*trajectories, strict=True))
            merged = [np.stack(arrays, axis=0) for arrays in transposed]

            with self.buffer_lock:
                self.buffer.add(*merged)

    def training(self):
        while True:
            with self.buffer_lock:
                can_sample = self.buffer.can_sample()
            if can_sample:
                self.logger.info("Start training.")
                break
            else:
                self.logger.info("Waiting.")
                time.sleep(1)

        transitions_processed = 0
        for training_step in itertools.count(start=1, step=1):
            with self.buffer_lock:
                sample = self.buffer.sample()

            data = sample["experience"]
            length_batch, length_traj = data["obs"].shape[:2]
            mask_done = jnp.logical_or(data["truncated"] == 1, data["terminated"] == 1)
            indices_done = jnp.argmax(mask_done, axis=1)
            has_one = jnp.any(mask_done, axis=1)
            indices_done = jnp.where(has_one, indices_done, length_traj - 1)
            batch_indices = jnp.arange(length_batch)

            obs = data["obs"][:, 0, ...]
            next_obs = data["obs"][batch_indices, indices_done]
            actions = data["action"][:, 0]

            indices = jnp.arange(length_traj)
            mask = indices[None, :] < indices_done[:, None]
            masked_rewards = data["reward"] * mask
            weighted_rewards = self._cumulative_discount_vector * masked_rewards
            rewards = jnp.sum(weighted_rewards, axis=1)

            terminals = data["terminated"][batch_indices, indices_done].astype(jnp.float32)

            self.flax_state, loss = train(
                self.flax_state,
                obs,
                next_obs,
                actions,
                rewards,
                terminals,
                self.cumulative_gamma,
                self.loss_type,
            )
            transitions_processed += self.batch_size

            if training_step == self.training_steps:
                self.is_done = True
                ray.get(self.controller.set_done.remote())
                self.logger.info(f"Final training step {training_step} reached; finishing.")
                break

            if training_step % self.summary_writing_period == 0:
                store_size = ray.get(self.controller.store_size.remote())
                self.logger.info(f"Training step: {training_step}.")
                self.logger.debug(f"Weights store size: {store_size}.")
                self.logger.info(f"Transitions processed by the trainer: {transitions_processed}.")
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
