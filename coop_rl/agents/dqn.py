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

import itertools
import logging
import os
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import ml_collections
import optax
import orbax.checkpoint as ocp
import ray
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample
from flax import core, struct
from flax.core.frozen_dict import FrozenDict
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.metrics import tensorboard
from flax.training import train_state
from typing_extensions import NamedTuple

from coop_rl.base_types import (
    ActorApply,
)
from coop_rl.buffers import TimeStep
from coop_rl.loss import q_learning
from coop_rl.multistep import batch_discounted_returns
from coop_rl.workers.auxiliary import BufferKeeper


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    info: dict


class TrainState(train_state.TrainState):
    key: jax.Array
    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tau: int  # smoothing coefficient for target networks

    def apply_gradients(self, *, grads, **kwargs):
        """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

        Note that internally this function calls ``.tx.update()`` followed by a call
        to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

        Args:
          grads: Gradients that have the same pytree structure as ``.params``.
          **kwargs: Additional dataclass attributes that should be ``.replace()``-ed.

        Returns:
          An updated instance of ``self`` with ``step`` incremented by one, ``params``
          and ``opt_state`` updated by applying ``grads``, and additional attributes
          replaced as specified by ``kwargs``.
        """
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads["params"]
            params_with_opt = self.params["params"]
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        # UPDATE Q PARAMS AND OPTIMISER STATE
        updates, new_opt_state = self.tx.update(grads_with_opt, self.opt_state, params_with_opt)
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)
        new_target_params = optax.incremental_update(new_params_with_opt, self.target_params, self.tau)

        # As implied by the OWG name, the gradients are used directly to update the
        # parameters.
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                "params": new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            target_params=new_target_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def create_train_state(rng, network, args_network, optimizer, args_optimizer, obs_shape, tau):
    state_rng, init_rng = jax.random.split(rng)
    model = network(**args_network)
    params = model.init(init_rng, jnp.ones((1, *obs_shape)))
    tx = optimizer(**args_optimizer)
    return TrainState.create(apply_fn=model.apply, params=params, target_params=params, key=state_rng, tx=tx, tau=tau)


def restore_dqn_flax_state(num_actions, network, optimizer, observation_shape, learning_rate, eps, checkpointdir):
    orbax_checkpointer = ocp.StandardCheckpointer()
    args_network = {"num_actions": num_actions}
    args_optimizer = {"learning_rate": learning_rate, "eps": eps}
    rng = jax.random.PRNGKey(0)  # jax.random.key(0)
    state = create_train_state(rng, network, args_network, optimizer, args_optimizer, observation_shape)
    abstract_my_tree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    return orbax_checkpointer.restore(checkpointdir, args=ocp.args.StandardRestore(abstract_my_tree))


def get_update_step(q_apply_fn: ActorApply, config: ml_collections.ConfigDict) -> Callable:
    def _update_step(train_state: TrainState, buffer_sample: TrajectoryBufferSample) -> tuple[TrainState, dict]:
        def _q_loss_fn(
            q_params: FrozenDict,
            target_q_params: FrozenDict,
            transitions: Transition,
        ) -> tuple[jnp.ndarray, dict]:
            q_tm1 = q_apply_fn(q_params, transitions.obs).preferences
            q_t = q_apply_fn(target_q_params, transitions.next_obs).preferences

            # Cast and clip rewards.
            discount = 1.0 - transitions.done.astype(jnp.float32)
            d_t = (discount * config.gamma).astype(jnp.float32)
            r_t = jnp.clip(transitions.reward, -config.max_abs_reward, config.max_abs_reward).astype(jnp.float32)
            a_tm1 = transitions.action

            # Compute Q-learning loss.
            batch_loss = q_learning(
                q_tm1,
                a_tm1,
                r_t,
                d_t,
                q_t,
                config.huber_loss_parameter,
            )

            loss_info = {
                "q_loss": batch_loss,
            }

            return batch_loss, loss_info

        sample: TimeStep = buffer_sample.experience
        # Extract the first and last observations.
        step_0_obs = jax.tree_util.tree_map(lambda x: x[:, 0], sample).obs
        step_0_actions = sample.action[:, 0]
        step_n_obs = jax.tree_util.tree_map(lambda x: x[:, -1], sample).obs
        # check if any of the transitions are done - this will be used to decide
        # if bootstrapping is needed
        done = jnp.logical_or(sample.truncated == 1, sample.terminated == 1)
        n_step_done = jnp.any(done, axis=-1)
        # Calculate the n-step rewards and select the first one.
        discounts = 1.0 - done.astype(jnp.float32)
        n_step_reward = batch_discounted_returns(
            sample.reward,
            discounts * config.gamma,
            jnp.zeros_like(discounts),
        )[:, 0]
        transitions = Transition(
            obs=step_0_obs,
            action=step_0_actions,
            reward=n_step_reward,
            done=n_step_done,
            next_obs=step_n_obs,
            info={},
        )

        # CALCULATE Q LOSS
        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            train_state.params,
            train_state.target_params,
            transitions,
        )
        train_state = train_state.apply_gradients(grads=q_grads)

        # PACK LOSS INFO
        loss_info = {
            **q_loss_info,
        }

        return train_state, loss_info

    return _update_step


def get_update_epoch(update_step_fn: Callable) -> Callable:
    @jax.jit
    def _update_epoch(train_state: TrainState, samples: list[TrajectoryBufferSample]):
            for sample in samples:
                train_state, loss_info = update_step_fn(train_state, sample)
            return train_state, loss_info
    return _update_epoch


class DQN(BufferKeeper):
    def __init__(
        self,
        *,
        trainer_seed,
        log_level,
        workdir,
        steps,
        training_iterations_per_step,
        summary_writing_period,
        save_period,
        synchronization_period,
        observation_shape,
        flax_state,
        dqn_params,
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
        self.dqn_params = dqn_params

        self.synchronization_period = synchronization_period
        self.summary_writing_period = summary_writing_period
        self.save_period = save_period
        self.summary_writer = tensorboard.SummaryWriter(os.path.join(workdir, "tensorboard/"))
        self.orbax_checkpointer = ocp.StandardCheckpointer()

        self._rng = jax.random.PRNGKey(trainer_seed)
        if flax_state is None:
            self._rng, rng = jax.random.split(self._rng)
            self.flax_state = create_train_state(
                rng,
                network,
                args_network,
                optimizer,
                args_optimizer,
                observation_shape,
                dqn_params.tau,
            )
        else:
            self.flax_state = flax_state

        self.controller = controller
        self.futures = self.controller.set_parameters.remote(self.flax_state.params)
        self.is_done = False

    def training(self):
        update_epoch_fn = get_update_epoch(get_update_step(self.flax_state.apply_fn, self.dqn_params))
        samples_generator = self.get_samples(self.training_iterations_per_step)
        transitions_processed = 0
        for step in itertools.count(start=1, step=1):
            samples = next(samples_generator)
            self.flax_state, loss_info = update_epoch_fn(self.flax_state, samples)
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
                self.summary_writer.scalar("loss", loss_info["q_loss"], self.flax_state.step)
                self.summary_writer.flush()

            if step % self.synchronization_period == 0:
                ray.get(self.futures)
                self.futures = self.controller.set_parameters.remote(self.flax_state.params)

            if step % self.save_period == 0:
                orbax_checkpoint_path = os.path.join(self.workdir, f"chkpt_train_step_{self.flax_state.step:07}")
                self.orbax_checkpointer.save(orbax_checkpoint_path, self.flax_state)
                self.logger.info(f"Orbax checkpoint is in: {orbax_checkpoint_path}")
