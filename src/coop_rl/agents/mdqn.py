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
    TimeStepDQNRecurrent,
)
from coop_rl.base.buffers import TimeStepDQN
from coop_rl.base.loss import munchausen_q_learning, munchausen_q_learning_n_step
from coop_rl.base.multistep import batch_discounted_returns
from coop_rl.networks.base import ScannedRNN


class RecurrentRolloutSample(NamedTuple):
    q_online: chex.Array  # (batch, learn_length, num_actions)
    q_target: chex.Array  # (batch, learn_length, num_actions)
    action: chex.Array  # (batch, learn_length)
    reward: chex.Array  # (batch, learn_length)
    terminated: chex.Array  # (batch, learn_length)
    truncated: chex.Array  # (batch, learn_length)


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


def create_train_state(rng, network, args_network, optimizer, args_optimizer, obs_shape, tau):
    state_rng, init_rng = jax.random.split(rng)
    model = network(**args_network)
    params = model.init(init_rng, jnp.ones((1, *obs_shape)))
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


def create_recurrent_train_state(
    rng,
    network,
    args_network,
    optimizer,
    args_optimizer,
    obs_shape,
    hidden_state_dim,
    cell_type,
    tau,
):
    state_rng, init_rng = jax.random.split(rng)
    model = network(**args_network)
    dummy_hidden_state = ScannedRNN(hidden_state_dim, cell_type).initialize_carry(1)
    dummy_obs = jnp.ones((1, 1, *obs_shape))
    dummy_prev_action = jnp.zeros((1, 1), dtype=jnp.int32)
    dummy_prev_reward = jnp.zeros((1, 1))
    dummy_reset = jnp.zeros((1, 1), dtype=bool)
    params = model.init(
        init_rng, dummy_hidden_state, (dummy_obs, dummy_prev_action, dummy_prev_reward, dummy_reset)
    )
    tx = optimizer(**args_optimizer)
    return TrainState.create(
        apply_fn=model.apply, params=params, target_params=params, key=state_rng, tx=tx, tau=tau
    )


def restore_recurrent_dqn_flax_state(
    *,
    rng,
    network,
    args_network,
    optimizer,
    args_optimizer,
    observation_shape,
    hidden_state_dim,
    cell_type,
    tau,
    checkpointdir,
):
    state = create_recurrent_train_state(
        rng,
        network,
        args_network,
        optimizer,
        args_optimizer,
        observation_shape,
        hidden_state_dim,
        cell_type,
        tau,
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
        key, policy_key = jax.random.split(key)
        actor_policy = apply_fn(params, jnp.expand_dims(_preprocess(observation), axis=0))
        return key, actor_policy.sample(seed=policy_key)

    return select_action


def get_select_action_batch_fn(
    apply_fn: ActorApply, obs_preprocess_fn: Callable | None = None
) -> Callable:
    """Like get_select_action_fn but for a batch of N observations (num_envs, *obs_shape)."""
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    @jax.jit
    def select_action_batch(key, params, observations):
        key, policy_key = jax.random.split(key)
        actor_policy = apply_fn(params, _preprocess(observations))
        return key, actor_policy.sample(seed=policy_key)

    return select_action_batch


def get_select_action_recurrent_batch_fn(
    apply_fn: ActorApply, max_abs_reward: float, obs_preprocess_fn: Callable | None = None
) -> Callable:
    """Like get_select_action_batch_fn but threads a recurrent hidden state across calls."""
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    @jax.jit
    def select_action(
        key, params, hidden_state, observations, reset_mask, prev_action, prev_reward
    ):
        key, policy_key = jax.random.split(key)
        obs_t = jnp.expand_dims(_preprocess(observations), axis=0)
        reset_t = jnp.expand_dims(reset_mask, axis=0).astype(bool)
        prev_action_t = jnp.expand_dims(prev_action, axis=0)
        prev_reward_t = jnp.clip(
            jnp.expand_dims(prev_reward, axis=0).astype(jnp.float32),
            -max_abs_reward,
            max_abs_reward,
        )
        new_hidden_state, actor_policy = apply_fn(
            params, hidden_state, (obs_t, prev_action_t, prev_reward_t, reset_t)
        )
        action = actor_policy.sample(seed=policy_key)
        action = jnp.squeeze(action, axis=0)
        return key, new_hidden_state, action

    return select_action


def get_update_step(
    *,
    apply_fn: ActorApply,
    gamma: float,
    entropy_temperature: float,
    munchausen_coefficient: float,
    clip_value_min: float,
    huber_loss_parameter: float,
    max_abs_reward: float,
    obs_preprocess_fn: Callable | None = None,
) -> Callable:
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    def _update_step(
        train_state: TrainState, buffer_sample: TrajectoryBufferSample
    ) -> tuple[TrainState, dict]:
        def _q_loss_fn(
            q_params: FrozenDict,
            target_q_params: FrozenDict,
            sample: TimeStepDQN,
        ) -> tuple[jnp.ndarray, dict]:
            obs = _preprocess(sample.obs)
            q_tm1 = apply_fn(q_params, obs[:, 0]).preferences.astype(jnp.float32)
            q_target_seq = apply_fn(target_q_params, obs).preferences.astype(jnp.float32)

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

            q_tm1_target = q_target_seq[:, 0]
            q_t_target = q_target_seq[batch_indices, indices_done]
            a_tm1 = sample.action[:, 0]

            # Munchausen reward shaping for the intermediate steps: each reward gets
            # its own alpha * clip(tau * ln pi(a|s), l0, 0) bonus. Step 0's bonus is
            # added inside munchausen_q_learning.
            log_pi = entropy_temperature * jax.nn.log_softmax(
                q_target_seq / entropy_temperature, axis=-1
            )
            action_one_hot = jax.nn.one_hot(sample.action, q_target_seq.shape[-1])
            munchausen_bonus = jnp.clip(
                jnp.sum(action_one_hot * log_pi, axis=-1), clip_value_min, 0.0
            )
            step_positions = jnp.arange(length_traj)[jnp.newaxis, :]
            r_seq = jnp.clip(sample.reward.astype(jnp.float32), -max_abs_reward, max_abs_reward)
            r_seq = r_seq + munchausen_coefficient * munchausen_bonus * (
                step_positions >= 1
            ).astype(jnp.float32)
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
            d_t = (1.0 - terminated_at_cut.astype(jnp.float32)) * gamma**indices_done

            batch_loss = munchausen_q_learning(
                q_tm1,
                q_tm1_target,
                a_tm1,
                n_step_reward,
                d_t,
                q_t_target,
                entropy_temperature,
                munchausen_coefficient,
                clip_value_min,
                huber_loss_parameter,
            )

            loss_info = {
                "loss": batch_loss,
            }

            return batch_loss, loss_info

        sample: TimeStepDQN = buffer_sample.experience

        # CALCULATE Q LOSS
        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            train_state.params,
            train_state.target_params,
            sample,
        )
        train_state = train_state.apply_gradients(grads=q_grads)

        # PACK LOSS INFO
        loss_info = {
            **q_loss_info,
        }

        return train_state, loss_info

    return _update_step


def get_recurrent_rollout(
    *,
    apply_fn: ActorApply,
    burn_in_length: int,
    max_abs_reward: float,
    obs_preprocess_fn: Callable | None = None,
) -> Callable:
    """Rollout half of the recurrent DQN step: RNN forward passes only, no loss math.

    Warms up the hidden state from the sequence's stored starting state via a
    stop-gradient burn-in (run separately for online and target params), then
    unrolls the remaining "learn" steps to produce per-timestep Q-value sequences.
    """
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    def _recurrent_rollout(
        q_params: FrozenDict, target_q_params: FrozenDict, sample: TimeStepDQNRecurrent
    ) -> RecurrentRolloutSample:
        # (batch, T, ...) -> (T, batch, ...) to match ScannedRNN's time-major scan.
        sample_tm = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), sample)

        init_hidden_state = sample_tm.hidden_state[0]
        reset_tm = sample_tm.reset_hidden_state.astype(bool)
        prev_reward_tm = jnp.clip(
            sample_tm.prev_reward.astype(jnp.float32), -max_abs_reward, max_abs_reward
        )

        burn_obs = _preprocess(sample_tm.obs[:burn_in_length])
        burn_prev_action = sample_tm.prev_action[:burn_in_length]
        burn_prev_reward = prev_reward_tm[:burn_in_length]
        burn_reset = reset_tm[:burn_in_length]
        burn_input = (burn_obs, burn_prev_action, burn_prev_reward, burn_reset)
        learn_obs = _preprocess(sample_tm.obs[burn_in_length:])
        learn_prev_action = sample_tm.prev_action[burn_in_length:]
        learn_prev_reward = prev_reward_tm[burn_in_length:]
        learn_reset = reset_tm[burn_in_length:]
        learn_input = (learn_obs, learn_prev_action, learn_prev_reward, learn_reset)

        if burn_in_length > 0:
            online_hidden_state, _ = apply_fn(q_params, init_hidden_state, burn_input)
            online_hidden_state = jax.lax.stop_gradient(online_hidden_state)
            target_hidden_state, _ = apply_fn(target_q_params, init_hidden_state, burn_input)
            target_hidden_state = jax.lax.stop_gradient(target_hidden_state)
        else:
            online_hidden_state = init_hidden_state
            target_hidden_state = init_hidden_state

        _, online_pi = apply_fn(q_params, online_hidden_state, learn_input)
        _, target_pi = apply_fn(target_q_params, target_hidden_state, learn_input)

        q_online = jnp.swapaxes(online_pi.preferences, 0, 1).astype(jnp.float32)
        q_target = jnp.swapaxes(target_pi.preferences, 0, 1).astype(jnp.float32)

        return RecurrentRolloutSample(
            q_online=q_online,
            q_target=q_target,
            action=sample.action[:, burn_in_length:],
            reward=sample.reward[:, burn_in_length:],
            terminated=sample.terminated[:, burn_in_length:],
            truncated=sample.truncated[:, burn_in_length:],
        )

    return _recurrent_rollout


def get_update_step_recurrent(
    *,
    apply_fn: ActorApply,
    burn_in_length: int,
    n_steps: int,
    gamma: float,
    entropy_temperature: float,
    munchausen_coefficient: float,
    clip_value_min: float,
    huber_loss_parameter: float,
    max_abs_reward: float,
    obs_preprocess_fn: Callable | None = None,
    recurrent_rollout_fn: Callable | None = None,
) -> Callable:
    """DQN-step half of the recurrent DQN step: consumes a rollout's Q-value sequences.

    Computes an n-step Munchausen TD error at every learn-window timestep that
    has n successor steps inside the window (R2D2-style), sourcing the Q-value
    sequences from a recurrent_rollout_fn. Each sequence yields
    learn_length - n_steps TD errors; truncated anchors are masked out.
    """
    assert n_steps >= 1, "n_steps must be at least 1"
    _rollout_fn = recurrent_rollout_fn or get_recurrent_rollout(
        apply_fn=apply_fn,
        burn_in_length=burn_in_length,
        max_abs_reward=max_abs_reward,
        obs_preprocess_fn=obs_preprocess_fn,
    )

    def _update_step_recurrent(
        train_state: TrainState, buffer_sample: TrajectoryBufferSample
    ) -> tuple[TrainState, dict]:
        def _q_loss_fn(
            q_params: FrozenDict,
            target_q_params: FrozenDict,
            sample: TimeStepDQNRecurrent,
        ) -> tuple[jnp.ndarray, dict]:
            rollout = _rollout_fn(q_params, target_q_params, sample)

            r_t = jnp.clip(rollout.reward.astype(jnp.float32), -max_abs_reward, max_abs_reward)

            batch_loss = munchausen_q_learning_n_step(
                rollout.q_online,
                rollout.q_target,
                rollout.action,
                r_t,
                rollout.terminated,
                rollout.truncated,
                gamma,
                n_steps,
                entropy_temperature,
                munchausen_coefficient,
                clip_value_min,
                huber_loss_parameter,
            )

            loss_info = {
                "loss": batch_loss,
            }

            return batch_loss, loss_info

        sample: TimeStepDQNRecurrent = buffer_sample.experience

        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            train_state.params,
            train_state.target_params,
            sample,
        )
        train_state = train_state.apply_gradients(grads=q_grads)

        loss_info = {
            **q_loss_info,
        }

        return train_state, loss_info

    return _update_step_recurrent


def get_update_epoch(*, update_step_fn: Callable, **kwargs) -> Callable:
    @jax.jit
    def _update_epoch(train_state: TrainState, samples: list[TrajectoryBufferSample]):
        for sample in samples:
            train_state, loss_info = update_step_fn(train_state, sample)
        return train_state, loss_info

    return _update_epoch
