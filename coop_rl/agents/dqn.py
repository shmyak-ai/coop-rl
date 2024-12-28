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
import os
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import ray
from flax import core, struct
from flax.metrics import tensorboard
from flax.training import train_state

from coop_rl.workers.auxiliary import BufferKeeper


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
    params = model.init(init_rng, jnp.ones((1, *obs_shape)))
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


def _target_q(state, observations, rewards, terminals, cumulative_gamma):
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
def _train(
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

    target = _target_q(state, next_observations, rewards, terminals, cumulative_gamma)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params, target)
    state = state.apply_gradients(grads=grad)
    return state, loss


@functools.partial(jax.jit, static_argnums=(3,))
def train(state, batched_timesteps, cumulative_discount_vector, loss_type):
    length_batch, length_traj = batched_timesteps.obs.shape[:2]
    mask_done = jnp.logical_or(batched_timesteps.truncated == 1, batched_timesteps.terminated == 1)
    indices_done = jnp.argmax(mask_done, axis=1)
    has_one = jnp.any(mask_done, axis=1)
    indices_done = jnp.where(has_one, indices_done, length_traj - 1)
    batch_indices = jnp.arange(length_batch)

    obs = batched_timesteps.obs[:, 0, ...]
    next_obs = batched_timesteps.obs[batch_indices, indices_done]
    actions = batched_timesteps.action[:, 0]

    indices = jnp.arange(length_traj)
    mask = indices[None, :] < indices_done[:, None]
    masked_rewards = batched_timesteps.reward * mask
    weighted_rewards = cumulative_discount_vector * masked_rewards
    rewards = jnp.sum(weighted_rewards, axis=1)

    terminals = batched_timesteps.terminated[batch_indices, indices_done].astype(jnp.float32)

    state, loss = _train(
        state,
        obs,
        next_obs,
        actions,
        rewards,
        terminals,
        cumulative_discount_vector[-1],
        loss_type,
    )
    return state, loss


class DQN(BufferKeeper):
    def __init__(
        self,
        *,
        trainer_seed,
        log_level,
        workdir,
        steps,
        training_iterations_per_step,
        gamma,
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
        super().__init__(buffer, args_buffer, training_iterations_per_step)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.workdir = workdir

        self.steps = steps
        self.training_iterations_per_step = training_iterations_per_step
        self.batch_size = args_buffer.sample_batch_size

        self.synchronization_period = synchronization_period
        self.summary_writing_period = summary_writing_period
        self.save_period = save_period
        self.summary_writer = tensorboard.SummaryWriter(os.path.join(workdir, "tensorboard/"))
        self.orbax_checkpointer = ocp.StandardCheckpointer()

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
        self.is_done = False

    def training(self):
        sampler = self.get_samples(self.training_iterations_per_step)
        transitions_processed = 0
        for step in itertools.count(start=1, step=1):
            samples = next(sampler)
            for sample in samples:
                self.flax_state, loss = train(
                    self.flax_state, sample["experience"], self._cumulative_discount_vector, self.loss_type
                )
            transitions_processed += self.batch_size

            if step == self.steps:
                self.is_done = True
                ray.get(self.controller.set_done.remote())
                self.logger.info(f"Final training step {step} reached; finishing.")
                break

            if step % self.summary_writing_period == 0:
                store_size = ray.get(self.controller.store_size.remote())
                self.logger.info(f"Step: {step}.")
                self.logger.debug(f"Weights store size: {store_size}.")
                self.logger.info(f"Transitions processed by the trainer: {transitions_processed}.")
                self.summary_writer.scalar("loss", loss, self.flax_state.step)
                self.summary_writer.flush()

            if step % self.target_update_period == 0:
                self.flax_state = self.flax_state.update_target_params()

            if step % self.synchronization_period == 0:
                ray.get(self.futures)
                self.futures = self.controller.set_parameters.remote(self.flax_state.params)

            if step % self.save_period == 0:
                orbax_checkpoint_path = os.path.join(self.workdir, f"chkpt_train_step_{self.flax_state.step:07}")
                self.orbax_checkpointer.save(orbax_checkpoint_path, self.flax_state)
                self.logger.info(f"Orbax checkpoint is in: {orbax_checkpoint_path}")
