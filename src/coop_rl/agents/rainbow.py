# Copyright 2025 The Coop RL Authors.
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

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample
from flax import core, struct
from flax.core.frozen_dict import FrozenDict
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training import train_state
from typing_extensions import NamedTuple

from coop_rl.base.base_types import (
    ActorApply,
)
from coop_rl.base.buffers import TimeStepDQN
from coop_rl.base.loss import categorical_double_q_learning
from coop_rl.base.multistep import batch_discounted_returns


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
        new_target_params = optax.incremental_update(
            new_params_with_opt, self.target_params, self.tau
        )

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

    def get_key(self):
        in_key, out_key = jax.random.split(self.key)
        return self.replace(key=in_key), out_key


def create_train_state(rng, network, args_network, optimizer, args_optimizer, obs_shape, tau):
    state_rng, init_rng, noise_rng = jax.random.split(rng, num=3)
    rngs = {"params": init_rng, "noise": noise_rng}
    model = network(**args_network)
    params = model.init(rngs, jnp.ones((1, *obs_shape)))
    tx = optimizer(**args_optimizer)
    return TrainState.create(
        apply_fn=model.apply, params=params, target_params=params, key=state_rng, tx=tx, tau=tau
    )


def restore_dqn_flax_state(
    *, rng, network, args_network, optimizer, args_optimizer, observation_shape, tau, checkpointdir
):
    state = create_train_state(
        rng, network, args_network, optimizer, args_optimizer, observation_shape, tau
    )
    if checkpointdir is None:
        return state
    orbax_checkpointer = ocp.StandardCheckpointer()
    abstract_my_tree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    return orbax_checkpointer.restore(checkpointdir, abstract_my_tree)


def get_select_action_fn(
    apply_fn: ActorApply, obs_preprocess_fn: Callable | None = None
) -> Callable:
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    @jax.jit
    def select_action(key, params, observation):
        key, noise_key, policy_key = jax.random.split(key, num=3)
        actor_policy, q_logits, atoms = apply_fn(
            params, jnp.expand_dims(_preprocess(observation), axis=0), rngs={"noise": noise_key}
        )
        return key, actor_policy.sample(seed=policy_key)

    return select_action


def get_select_action_batch_fn(
    apply_fn: ActorApply, obs_preprocess_fn: Callable | None = None
) -> Callable:
    """Like get_select_action_fn but for a batch of N observations (num_envs, *obs_shape)."""
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    @jax.jit
    def select_action_batch(key, params, observations):
        key, noise_key, policy_key = jax.random.split(key, num=3)
        actor_policy, q_logits, atoms = apply_fn(
            params, _preprocess(observations), rngs={"noise": noise_key}
        )
        return key, actor_policy.sample(seed=policy_key)

    return select_action_batch


def get_update_step(
    *,
    apply_fn: ActorApply,
    gamma: float,
    max_abs_reward: float,
    importance_weight_scheduler_fn: Callable,
    obs_preprocess_fn: Callable | None = None,
) -> Callable:
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    @jax.jit
    def _update_step(
        train_state: TrainState, buffer_sample: TrajectoryBufferSample
    ) -> tuple[TrainState, dict]:
        def _q_loss_fn(
            q_params: FrozenDict,
            target_q_params: FrozenDict,
            transitions: Transition,
            transition_probs: chex.Array,
            noise_key: chex.PRNGKey,
            importance_sampling_exponent: float,
        ) -> tuple[jnp.ndarray, dict]:
            noise_key_tm1, noise_key_t, noise_key_select = jax.random.split(noise_key, num=3)

            _, q_logits_tm1, q_atoms_tm1 = apply_fn(
                q_params, transitions.obs, rngs={"noise": noise_key_tm1}
            )
            q_logits_tm1 = q_logits_tm1.astype(jnp.float32)
            q_atoms_tm1 = q_atoms_tm1.astype(jnp.float32)
            _, q_logits_t, q_atoms_t = apply_fn(
                target_q_params, transitions.next_obs, rngs={"noise": noise_key_t}
            )
            q_logits_t = q_logits_t.astype(jnp.float32)
            q_atoms_t = q_atoms_t.astype(jnp.float32)
            q_t_selector_dist, _, _ = apply_fn(
                q_params, transitions.next_obs, rngs={"noise": noise_key_select}
            )
            q_t_selector = q_t_selector_dist.preferences

            # Cast and clip rewards.
            discount = 1.0 - transitions.done.astype(jnp.float32)
            d_t = (discount * gamma).astype(jnp.float32)
            r_t = jnp.clip(transitions.reward, -max_abs_reward, max_abs_reward).astype(jnp.float32)
            a_tm1 = transitions.action

            batch_q_error = categorical_double_q_learning(
                q_logits_tm1, q_atoms_tm1, a_tm1, r_t, d_t, q_logits_t, q_atoms_t, q_t_selector
            )
            batch_q_error = jnp.maximum(batch_q_error, 0)

            # Importance weighting.
            importance_weights = (1.0 / (transition_probs + 1e-10)).astype(jnp.float32)
            importance_weights **= importance_sampling_exponent
            importance_weights /= jnp.max(importance_weights)

            # Reweight.
            q_loss = jnp.mean(importance_weights * batch_q_error)
            new_priorities = jnp.sqrt(batch_q_error) + 1e-5

            loss_info = {
                "loss": q_loss,
                "priorities": new_priorities,
            }

            return q_loss, loss_info

        sample: TimeStepDQN = buffer_sample.experience

        # Get indices of the last observation
        length_batch, length_traj = sample.obs.shape[:2]
        mask_done = jnp.logical_or(sample.truncated == 1, sample.terminated == 1)
        indices_done = jnp.argmax(mask_done, axis=1)
        has_one = jnp.any(mask_done, axis=1)
        indices_done = jnp.where(has_one, indices_done, length_traj - 1)
        batch_indices = jnp.arange(length_batch)

        # Extract the first and last observations.
        step_0_obs = jax.tree_util.tree_map(lambda x: x[:, 0], sample).obs
        step_0_actions = sample.action[:, 0]
        step_n_obs = jax.tree_util.tree_map(lambda x: x[batch_indices, indices_done], sample).obs
        # check if any of the transitions are done - this will be used to decide
        # if bootstrapping is needed
        n_step_done = jnp.any(sample.terminated == 1, axis=-1)
        # Calculate the n-step rewards and select the first one.
        discounts = 1.0 - mask_done.astype(jnp.float32)
        n_step_reward = batch_discounted_returns(
            sample.reward.astype(jnp.float32),
            discounts * gamma,
            jnp.zeros_like(discounts),
        )[:, 0]
        transitions = Transition(
            obs=_preprocess(step_0_obs),
            action=step_0_actions,
            reward=n_step_reward,
            done=n_step_done,
            next_obs=_preprocess(step_n_obs),
            info={},
        )

        importance_sampling_exponent = importance_weight_scheduler_fn(train_state.step)
        train_state, noise_key = train_state.get_key()

        # CALCULATE Q LOSS
        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            train_state.params,
            train_state.target_params,
            transitions,
            buffer_sample.probabilities,
            noise_key,
            importance_sampling_exponent,
        )
        train_state = train_state.apply_gradients(grads=q_grads)

        # PACK LOSS INFO
        info = {
            **q_loss_info,
            "importance_sampling_exponent": importance_sampling_exponent,
        }

        return train_state, info

    # _checked_update_step = checkify.checkify(
    #     _update_step, errors=checkify.float_checks
    # )
    # return _checked_update_step
    return _update_step


def get_update_epoch(
    *, update_step_fn: Callable, buffer_lock=None, buffer=None, **kwargs
) -> Callable:
    def _update_epoch(train_state: TrainState, samples: list[TrajectoryBufferSample]):
        for sample in samples:
            train_state, info = update_step_fn(train_state, sample)
            if buffer_lock is not None and buffer is not None:
                with buffer_lock.write():
                    buffer.set_priorities(sample.indices, info["priorities"])
        return train_state, info

    return _update_epoch
