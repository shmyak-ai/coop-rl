# Copyright 2026 The Coop RL Authors.
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
from coop_rl.base.loss import munchausen_quantile_q_learning
from coop_rl.base.multistep import batch_discounted_returns


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    discount: chex.Array
    next_obs: chex.Array
    q_tm1_target: chex.Array
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
    state_rng, init_rng, quantile_rng = jax.random.split(rng, num=3)
    rngs = {"params": init_rng, "quantiles": quantile_rng}
    model = network(**args_network)
    # Parameter shapes do not depend on the number of quantile samples.
    params = model.init(rngs, jnp.ones((1, *obs_shape)), 1)
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
    apply_fn: ActorApply, num_quantile_samples: int, obs_preprocess_fn: Callable | None = None
) -> Callable:
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    @jax.jit
    def select_action(key, params, observation):
        key, quantile_key, policy_key = jax.random.split(key, num=3)
        actor_policy, _, _ = apply_fn(
            params,
            jnp.expand_dims(_preprocess(observation), axis=0),
            num_quantile_samples,
            rngs={"quantiles": quantile_key},
        )
        return key, actor_policy.sample(seed=policy_key)

    return select_action


def get_select_action_batch_fn(
    apply_fn: ActorApply, num_quantile_samples: int, obs_preprocess_fn: Callable | None = None
) -> Callable:
    """Like get_select_action_fn but for a batch of N observations (num_envs, *obs_shape)."""
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    @jax.jit
    def select_action_batch(key, params, observations):
        key, quantile_key, policy_key = jax.random.split(key, num=3)
        actor_policy, _, _ = apply_fn(
            params,
            _preprocess(observations),
            num_quantile_samples,
            rngs={"quantiles": quantile_key},
        )
        return key, actor_policy.sample(seed=policy_key)

    return select_action_batch


def get_update_step(
    *,
    apply_fn: ActorApply,
    gamma: float,
    entropy_temperature: float,
    munchausen_coefficient: float,
    clip_value_min: float,
    quantile_huber_kappa: float,
    num_tau_samples: int,
    num_tau_prime_samples: int,
    num_quantile_samples: int,
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
            quantile_key: chex.PRNGKey,
            importance_sampling_exponent: float,
        ) -> tuple[jnp.ndarray, dict]:
            online_key, target_key = jax.random.split(quantile_key)
            _, z_tm1, quantiles_tm1 = apply_fn(
                q_params, transitions.obs, num_tau_samples, rngs={"quantiles": online_key}
            )
            z_tm1 = z_tm1.astype(jnp.float32)
            quantiles_tm1 = quantiles_tm1.astype(jnp.float32)
            _, z_t_target, _ = apply_fn(
                target_q_params,
                transitions.next_obs,
                num_tau_prime_samples,
                rngs={"quantiles": target_key},
            )
            z_t_target = z_t_target.astype(jnp.float32)

            batch_q_error = munchausen_quantile_q_learning(
                z_tm1,
                quantiles_tm1,
                transitions.q_tm1_target,
                z_t_target,
                transitions.action,
                transitions.reward,
                transitions.discount,
                entropy_temperature,
                munchausen_coefficient,
                clip_value_min,
                quantile_huber_kappa,
            )

            # Importance weighting.
            importance_weights = (1.0 / (transition_probs + 1e-10)).astype(jnp.float32)
            importance_weights **= importance_sampling_exponent
            importance_weights /= jnp.max(importance_weights)

            # Reweight.
            q_loss = jnp.mean(importance_weights * batch_q_error)
            new_priorities = batch_q_error + 1e-5

            loss_info = {
                "loss": q_loss,
                "priorities": new_priorities,
            }

            return q_loss, loss_info

        sample: TimeStepDQN = buffer_sample.experience

        # The first done transition cuts the window: it caps the reward sum and
        # provides the bootstrap observation (default: the last transition).
        length_batch, length_traj = sample.action.shape[:2]
        mask_done = jnp.logical_or(sample.truncated == 1, sample.terminated == 1)
        indices_done = jnp.argmax(mask_done, axis=1)
        has_one = jnp.any(mask_done, axis=1)
        indices_done = jnp.where(has_one, indices_done, length_traj - 1)
        batch_indices = jnp.arange(length_batch)
        # Bootstrapping is skipped only if the cut transition itself terminated;
        # a truncated episode is still bootstrapped from its last observation.
        terminated_at_cut = sample.terminated[batch_indices, indices_done] == 1

        train_state, key = train_state.get_key()
        policy_key, loss_key = jax.random.split(key)

        # Munchausen reward shaping for the intermediate steps: each reward gets
        # its own alpha * clip(tau * ln pi(a|s), l0, 0) bonus, with pi computed from
        # the target network's mean-quantile q-values. Step 0's bonus is added
        # inside munchausen_quantile_q_learning from q_tm1_target.
        obs = _preprocess(sample.obs)
        _, z_target_seq, _ = apply_fn(
            train_state.target_params, obs, num_quantile_samples, rngs={"quantiles": policy_key}
        )
        q_target_seq = jnp.mean(z_target_seq, axis=-2).astype(jnp.float32)
        log_pi = entropy_temperature * jax.nn.log_softmax(
            q_target_seq / entropy_temperature, axis=-1
        )
        action_one_hot = jax.nn.one_hot(sample.action, q_target_seq.shape[-1])
        munchausen_bonus = jnp.clip(jnp.sum(action_one_hot * log_pi, axis=-1), clip_value_min, 0.0)
        step_positions = jnp.arange(length_traj)[jnp.newaxis, :]
        r_seq = jnp.clip(sample.reward.astype(jnp.float32), -max_abs_reward, max_abs_reward)
        r_seq = r_seq + munchausen_coefficient * munchausen_bonus * (step_positions >= 1).astype(
            jnp.float32
        )
        # The cut transition's reward is part of the bootstrap value, unless it
        # terminated the episode (then it is the final reward, with no bootstrap).
        cut_mask = (step_positions == indices_done[:, jnp.newaxis]) & ~terminated_at_cut[
            :, jnp.newaxis
        ]
        r_seq = jnp.where(cut_mask, 0.0, r_seq)

        discounts = 1.0 - mask_done.astype(jnp.float32)
        n_step_reward = batch_discounted_returns(
            r_seq,
            discounts * gamma,
            jnp.zeros_like(discounts),
        )[:, 0]
        # The bootstrap is n steps away from step 0, so it is discounted gamma**n.
        n_step_discount = (1.0 - terminated_at_cut.astype(jnp.float32)) * gamma**indices_done
        transitions = Transition(
            obs=jax.tree_util.tree_map(lambda x: x[:, 0], obs),
            action=sample.action[:, 0],
            reward=n_step_reward,
            discount=n_step_discount,
            next_obs=jax.tree_util.tree_map(lambda x: x[batch_indices, indices_done], obs),
            q_tm1_target=q_target_seq[:, 0],
            info={},
        )

        importance_sampling_exponent = importance_weight_scheduler_fn(train_state.step)

        # CALCULATE Q LOSS
        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            train_state.params,
            train_state.target_params,
            transitions,
            buffer_sample.probabilities,
            loss_key,
            importance_sampling_exponent,
        )
        train_state = train_state.apply_gradients(grads=q_grads)

        # PACK LOSS INFO
        info = {
            **q_loss_info,
            "importance_sampling_exponent": importance_sampling_exponent,
        }

        return train_state, info

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
