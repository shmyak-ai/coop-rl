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
    TimeStepDQNRecurrent,
)
from coop_rl.base.buffers import TimeStepDQN
from coop_rl.base.loss import (
    munchausen_quantile_q_learning,
    munchausen_quantile_q_learning_n_step,
)
from coop_rl.base.multistep import batch_discounted_returns
from coop_rl.networks.base import ScannedRNN


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    discount: chex.Array
    next_obs: chex.Array
    q_tm1_target: chex.Array
    info: dict


class RecurrentQuantileRolloutSample(NamedTuple):
    z_online: chex.Array  # (batch, learn_length, num_tau_samples, num_actions)
    quantiles_online: chex.Array  # (batch, learn_length, num_tau_samples)
    z_target: chex.Array  # (batch, learn_length, num_tau_prime_samples, num_actions)
    action: chex.Array  # (batch, learn_length)
    reward: chex.Array  # (batch, learn_length)
    terminated: chex.Array  # (batch, learn_length)
    truncated: chex.Array  # (batch, learn_length)


class TrainState(train_state.TrainState):
    key: jax.Array
    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tau: int  # smoothing coefficient for target networks
    # 0 (default): Polyak-blend the target every step via tau. >0: hard-copy the
    # target from the online params every target_update_period steps instead.
    target_update_period: int = struct.field(pytree_node=False, default=0)

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
        if self.target_update_period > 0:
            new_target_params = optax.periodic_update(
                new_params_with_opt, self.target_params, self.step + 1, self.target_update_period
            )
        else:
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


def create_train_state(
    rng, network, args_network, optimizer, args_optimizer, obs_shape, tau, target_update_period=0
):
    state_rng, init_rng, quantile_rng, noise_rng = jax.random.split(rng, num=4)
    rngs = {"params": init_rng, "quantiles": quantile_rng, "noise": noise_rng}
    model = network(**args_network)
    # Parameter shapes do not depend on the number of quantile samples.
    params = model.init(rngs, jnp.ones((1, *obs_shape)), 1)
    tx = optimizer(**args_optimizer)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        target_params=params,
        key=state_rng,
        tx=tx,
        tau=tau,
        target_update_period=target_update_period,
    )


def restore_dqn_flax_state(
    *,
    rng,
    network,
    args_network,
    optimizer,
    args_optimizer,
    observation_shape,
    tau,
    target_update_period=0,
    checkpointdir,
):
    state = create_train_state(
        rng,
        network,
        args_network,
        optimizer,
        args_optimizer,
        observation_shape,
        tau,
        target_update_period,
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
    state_rng, init_rng, quantile_rng = jax.random.split(rng, num=3)
    rngs = {"params": init_rng, "quantiles": quantile_rng}
    model = network(**args_network)
    dummy_hidden_state = ScannedRNN(hidden_state_dim, cell_type).initialize_carry(1)
    dummy_obs = jnp.ones((1, 1, *obs_shape))
    dummy_prev_action = jnp.zeros((1, 1), dtype=jnp.int32)
    dummy_prev_reward = jnp.zeros((1, 1))
    dummy_reset = jnp.zeros((1, 1), dtype=bool)
    # Parameter shapes do not depend on the number of quantile samples.
    params = model.init(
        rngs,
        dummy_hidden_state,
        (dummy_obs, dummy_prev_action, dummy_prev_reward, dummy_reset),
        1,
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
    apply_fn: ActorApply,
    num_quantile_samples: int,
    obs_preprocess_fn: Callable | None = None,
    epsilon_scheduler_fn: Callable[[int], float] | None = None,
) -> Callable:
    """Like get_select_action_fn but for a batch of N observations (num_envs, *obs_shape).

    `epsilon_scheduler_fn`, if given, is a step-indexed schedule (e.g. `optax.linear_schedule`)
    overriding the network's static epsilon; step count is tracked (in environment transitions)
    by a plain Python counter closed over here, since the collector always calls the returned
    function as `(key, params, observations)` regardless of agent.
    """
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    if epsilon_scheduler_fn is None:

        @jax.jit
        def select_action_batch(key, params, observations):
            key, quantile_key, noise_key, policy_key = jax.random.split(key, num=4)
            actor_policy, _, _ = apply_fn(
                params,
                _preprocess(observations),
                num_quantile_samples,
                rngs={"quantiles": quantile_key, "noise": noise_key},
            )
            return key, actor_policy.sample(seed=policy_key)

        return select_action_batch

    @jax.jit
    def _select_action_batch_eps(key, params, observations, epsilon):
        key, quantile_key, noise_key, policy_key = jax.random.split(key, num=4)
        actor_policy, _, _ = apply_fn(
            params,
            _preprocess(observations),
            num_quantile_samples,
            epsilon,
            rngs={"quantiles": quantile_key, "noise": noise_key},
        )
        return key, actor_policy.sample(seed=policy_key)

    step_count = 0

    def select_action_batch(key, params, observations):
        nonlocal step_count
        step_count += observations.shape[0]
        epsilon = epsilon_scheduler_fn(step_count)
        return _select_action_batch_eps(key, params, observations, epsilon)

    return select_action_batch


def get_select_action_recurrent_batch_fn(
    apply_fn: ActorApply,
    num_quantile_samples: int,
    max_abs_reward: float,
    obs_preprocess_fn: Callable | None = None,
) -> Callable:
    """Like get_select_action_batch_fn but threads a recurrent hidden state across calls."""
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    @jax.jit
    def select_action(
        key, params, hidden_state, observations, reset_mask, prev_action, prev_reward
    ):
        key, quantile_key, policy_key = jax.random.split(key, num=3)
        obs_t = jnp.expand_dims(_preprocess(observations), axis=0)
        reset_t = jnp.expand_dims(reset_mask, axis=0).astype(bool)
        prev_action_t = jnp.expand_dims(prev_action, axis=0)
        prev_reward_t = jnp.clip(
            jnp.expand_dims(prev_reward, axis=0).astype(jnp.float32),
            -max_abs_reward,
            max_abs_reward,
        )
        new_hidden_state, (actor_policy, _, _) = apply_fn(
            params,
            hidden_state,
            (obs_t, prev_action_t, prev_reward_t, reset_t),
            num_quantile_samples,
            rngs={"quantiles": quantile_key},
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
            online_key, target_key, online_noise_key, target_noise_key = jax.random.split(
                quantile_key, num=4
            )
            _, z_tm1, quantiles_tm1 = apply_fn(
                q_params,
                transitions.obs,
                num_tau_samples,
                rngs={"quantiles": online_key, "noise": online_noise_key},
            )
            z_tm1 = z_tm1.astype(jnp.float32)
            quantiles_tm1 = quantiles_tm1.astype(jnp.float32)
            _, z_t_target, _ = apply_fn(
                target_q_params,
                transitions.next_obs,
                num_tau_prime_samples,
                rngs={"quantiles": target_key, "noise": target_noise_key},
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
        policy_key, policy_noise_key, loss_key = jax.random.split(key, num=3)

        # Munchausen reward shaping for the intermediate steps: each reward gets
        # its own alpha * clip(tau * ln pi(a|s), l0, 0) bonus, with pi computed from
        # the target network's mean-quantile q-values. Step 0's bonus is added
        # inside munchausen_quantile_q_learning from q_tm1_target.
        obs = _preprocess(sample.obs)
        _, z_target_seq, _ = apply_fn(
            train_state.target_params,
            obs,
            num_quantile_samples,
            rngs={"quantiles": policy_key, "noise": policy_noise_key},
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


def get_recurrent_rollout(
    *,
    apply_fn: ActorApply,
    burn_in_length: int,
    num_tau_samples: int,
    num_tau_prime_samples: int,
    max_abs_reward: float,
    obs_preprocess_fn: Callable | None = None,
) -> Callable:
    """Rollout half of the recurrent M-IQN step: RNN forward passes only, no loss math.

    Warms up the hidden state from the sequence's stored starting state via a
    stop-gradient burn-in (run separately for online and target params), then
    unrolls the remaining "learn" steps to produce per-timestep quantile-value
    sequences (online with num_tau_samples, target with num_tau_prime_samples).
    """
    _preprocess = obs_preprocess_fn if obs_preprocess_fn is not None else lambda x: x

    def _recurrent_rollout(
        q_params: FrozenDict,
        target_q_params: FrozenDict,
        sample: TimeStepDQNRecurrent,
        quantile_key: chex.PRNGKey,
    ) -> RecurrentQuantileRolloutSample:
        burn_online_key, burn_target_key, online_key, target_key = jax.random.split(
            quantile_key, num=4
        )
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
            # The head output is discarded during burn-in, so one quantile sample.
            online_hidden_state, _ = apply_fn(
                q_params,
                init_hidden_state,
                burn_input,
                1,
                rngs={"quantiles": burn_online_key},
            )
            online_hidden_state = jax.lax.stop_gradient(online_hidden_state)
            target_hidden_state, _ = apply_fn(
                target_q_params,
                init_hidden_state,
                burn_input,
                1,
                rngs={"quantiles": burn_target_key},
            )
            target_hidden_state = jax.lax.stop_gradient(target_hidden_state)
        else:
            online_hidden_state = init_hidden_state
            target_hidden_state = init_hidden_state

        _, (_, z_online, quantiles_online) = apply_fn(
            q_params,
            online_hidden_state,
            learn_input,
            num_tau_samples,
            rngs={"quantiles": online_key},
        )
        _, (_, z_target, _) = apply_fn(
            target_q_params,
            target_hidden_state,
            learn_input,
            num_tau_prime_samples,
            rngs={"quantiles": target_key},
        )

        return RecurrentQuantileRolloutSample(
            z_online=jnp.swapaxes(z_online, 0, 1).astype(jnp.float32),
            quantiles_online=jnp.swapaxes(quantiles_online, 0, 1).astype(jnp.float32),
            z_target=jnp.swapaxes(z_target, 0, 1).astype(jnp.float32),
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
    quantile_huber_kappa: float,
    num_tau_samples: int,
    num_tau_prime_samples: int,
    max_abs_reward: float,
    obs_preprocess_fn: Callable | None = None,
    recurrent_rollout_fn: Callable | None = None,
) -> Callable:
    """M-IQN-step half of the recurrent step: consumes a rollout's quantile sequences.

    Computes an n-step Munchausen quantile TD error at every learn-window timestep
    that has n successor steps inside the window (R2D2-style), sourcing the quantile
    sequences from a recurrent_rollout_fn. Each sequence yields
    learn_length - n_steps anchors; truncated anchors are masked out. Uniform
    replay: no importance weights, no priorities.
    """
    assert n_steps >= 1, "n_steps must be at least 1"
    _rollout_fn = recurrent_rollout_fn or get_recurrent_rollout(
        apply_fn=apply_fn,
        burn_in_length=burn_in_length,
        num_tau_samples=num_tau_samples,
        num_tau_prime_samples=num_tau_prime_samples,
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
            quantile_key: chex.PRNGKey,
        ) -> tuple[jnp.ndarray, dict]:
            rollout = _rollout_fn(q_params, target_q_params, sample, quantile_key)

            r_t = jnp.clip(rollout.reward.astype(jnp.float32), -max_abs_reward, max_abs_reward)

            batch_loss = munchausen_quantile_q_learning_n_step(
                rollout.z_online,
                rollout.quantiles_online,
                rollout.z_target,
                rollout.action,
                r_t,
                rollout.terminated,
                rollout.truncated,
                gamma,
                n_steps,
                entropy_temperature,
                munchausen_coefficient,
                clip_value_min,
                quantile_huber_kappa,
            )

            loss_info = {
                "loss": batch_loss,
            }

            return batch_loss, loss_info

        sample: TimeStepDQNRecurrent = buffer_sample.experience

        train_state, quantile_key = train_state.get_key()

        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            train_state.params,
            train_state.target_params,
            sample,
            quantile_key,
        )
        train_state = train_state.apply_gradients(grads=q_grads)

        loss_info = {
            **q_loss_info,
        }

        return train_state, loss_info

    return _update_step_recurrent


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
