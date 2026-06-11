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
import threading

import flashbax as fbx
import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample

from coop_rl.agents.dreamer import ext_space as dreamer_ext_space
from coop_rl.base.base_types import TimeStepDQN


def _sample_with_indices(state, rng_key, batch_size, sequence_length, period):
    """Mirror flashbax trajectory_buffer.sample but also return the sampled slot indices.

    Returns (sample, batch_indices, time_indices) so the trainer can later scatter
    refreshed latent states back into exactly the slots that were drawn.
    """
    leaf = jax.tree_util.tree_leaves(state.experience)[0]
    add_batch_size, max_length_time_axis = leaf.shape[0], leaf.shape[1]

    max_time = jnp.where(state.is_full, max_length_time_axis, state.current_index)
    head = jnp.where(state.is_full, state.current_index, 0)
    max_start = max_time - sequence_length
    num_valid_items = jnp.where(max_start >= 0, (max_start // period) + 1, 0)

    rng_key, subkey_items = jax.random.split(rng_key)
    rng_key, subkey_batch = jax.random.split(rng_key)
    sampled_item_idx = jax.random.randint(subkey_items, (batch_size,), 0, num_valid_items)
    logical_start = sampled_item_idx * period
    physical_start = (head + logical_start) % max_length_time_axis
    batch_indices = jax.random.randint(subkey_batch, (batch_size,), 0, add_batch_size)
    time_indices = (physical_start[:, None] + jnp.arange(sequence_length)) % max_length_time_axis

    experience = jax.tree.map(lambda x: x[batch_indices[:, None], time_indices], state.experience)
    return TrajectoryBufferSample(experience=experience), batch_indices, time_indices


def _update_latents(state, batch_indices, time_indices, replay_updates):
    """Scatter refreshed latent entries back into their original buffer slots.

    Only keys present in both ``replay_updates`` and the stored experience are written
    (i.e. ``dyn/deter`` / ``dyn/stoch``); ``stepid`` is used solely as a guard. A slot is
    written only if its stored ``stepid`` still matches the sampled one, so trajectories
    overwritten by the ring buffer between sampling and write-back are left untouched.

    The replay payload covers only the trained steps: with ``replay_context = K`` the first
    K steps of each sampled window are consumed as warm-start context, so the entries span
    the last ``T - K`` steps. We therefore align ``time_indices`` to the payload from the right.
    """
    rows = batch_indices[:, None]
    time_indices = time_indices[:, -replay_updates["stepid"].shape[1] :]
    stored_stepid = state.experience["stepid"][rows, time_indices]
    match = jnp.all(stored_stepid == replay_updates["stepid"], axis=-1)  # (B, T)

    experience = dict(state.experience)
    for key, new_val in replay_updates.items():
        if key == "stepid" or key not in experience:
            continue
        old_val = experience[key][rows, time_indices]
        mask = jnp.expand_dims(match, axis=tuple(range(match.ndim, new_val.ndim)))
        merged = jnp.where(mask, new_val, old_val)
        experience[key] = experience[key].at[rows, time_indices].set(merged)
    return state.replace(experience=experience)


class BufferFlat:
    def __init__(
        self,
        buffer_seed,
        max_length,
        min_length,
        sample_batch_size,
        add_sequences,
        add_batch_size,
        observation_shape,
    ):
        self.buffer = fbx.make_flat_buffer(
            max_length, min_length, sample_batch_size, add_sequences, add_batch_size
        )
        fake_timestep = {"obs": jnp.array(observation_shape), "reward": jnp.array(1.0)}
        self.state = self.buffer.init(fake_timestep)
        self.rng_key = jax.random.PRNGKey(buffer_seed)

    def add(self, traj_batch):
        self.state = self.buffer.add(self.state, traj_batch)

    def sample(self):
        self.rng_key, rng_key = jax.random.split(self.rng_key)
        batch = self.buffer.sample(self.state, rng_key)
        return batch


class BufferTrajectoryDQN:
    def __init__(
        self,
        buffer_seed,
        add_batch_size,
        sample_batch_size,
        sample_sequence_length,
        period,
        min_length,
        max_size,
        observation_shape,
        time_step_dtypes,
    ):
        self.dtypes = time_step_dtypes()
        self.cpu = jax.devices("cpu")[0]
        with jax.default_device(self.cpu):
            self.buffer = fbx.make_trajectory_buffer(
                add_batch_size=add_batch_size,
                sample_batch_size=sample_batch_size,
                sample_sequence_length=sample_sequence_length,
                period=period,
                min_length_time_axis=min_length,
                max_length_time_axis=max_size // add_batch_size,
            )
            self.buffer = self.buffer.replace(
                init=jax.jit(self.buffer.init),
                add=jax.jit(self.buffer.add, donate_argnums=0),
                sample=jax.jit(self.buffer.sample),
                can_sample=jax.jit(self.buffer.can_sample),
            )
            fake_timestep = TimeStepDQN(
                obs=jnp.ones(observation_shape, dtype=self.dtypes.obs),
                action=jnp.ones((), dtype=self.dtypes.action),
                reward=jnp.ones((), dtype=self.dtypes.reward),
                terminated=jnp.ones((), dtype=self.dtypes.terminated),
                truncated=jnp.ones((), dtype=self.dtypes.truncated),
            )
            self.state = self.buffer.init(fake_timestep)
            self.rng_key = jax.random.PRNGKey(buffer_seed)
            self._rng_lock = threading.Lock()

    def add(self, batch_sequence):
        with jax.default_device(self.cpu):
            self.state = self.buffer.add(self.state, batch_sequence)

    def sample(self):
        with self._rng_lock:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        with jax.default_device(self.cpu):
            batch = self.buffer.sample(self.state, rng_key)
        return batch

    def can_sample(self):
        return self.buffer.can_sample(self.state)


class BufferTrajectoryDreamer:
    def __init__(
        self,
        args_network,
        buffer_seed,
        add_batch_size,
        sample_batch_size,
        sample_sequence_length,
        period,
        min_length,
        max_size,
        observation_shape,
        actions_shape,
    ):
        ext_space = dreamer_ext_space(args_network, observation_shape, actions_shape)
        self.cpu = jax.devices("cpu")[0]
        with jax.default_device(self.cpu):
            self.buffer = fbx.make_trajectory_buffer(
                add_batch_size=add_batch_size,
                sample_batch_size=sample_batch_size,
                sample_sequence_length=sample_sequence_length,
                period=period,
                min_length_time_axis=min_length,
                max_length_time_axis=max_size // add_batch_size,
            )
            self.buffer = self.buffer.replace(
                init=jax.jit(self.buffer.init),
                add=jax.jit(self.buffer.add, donate_argnums=0),
                sample=jax.jit(self.buffer.sample),
                can_sample=jax.jit(self.buffer.can_sample),
            )
            self._sample_with_indices = jax.jit(
                functools.partial(
                    _sample_with_indices,
                    batch_size=sample_batch_size,
                    sequence_length=sample_sequence_length,
                    period=period,
                )
            )
            self._update_latents = jax.jit(_update_latents, donate_argnums=0)
            self.dummy_timestep = {
                "image": jnp.ones(
                    observation_shape["image"].shape, dtype=observation_shape["image"].dtype
                ),
                "is_first": jnp.ones(
                    observation_shape["is_first"].shape, dtype=observation_shape["is_first"].dtype
                ),
                "is_last": jnp.ones(
                    observation_shape["is_last"].shape, dtype=observation_shape["is_last"].dtype
                ),
                "is_terminal": jnp.ones(
                    observation_shape["is_terminal"].shape,
                    dtype=observation_shape["is_terminal"].dtype,
                ),
                "reward": jnp.ones(
                    observation_shape["reward"].shape, dtype=observation_shape["reward"].dtype
                ),
                "stepid": jnp.ones(ext_space["stepid"].shape, dtype=ext_space["stepid"].dtype),
                "consec": jnp.ones(ext_space["consec"].shape, dtype=ext_space["consec"].dtype),
                "dyn/deter": jnp.ones(
                    ext_space["dyn/deter"].shape, dtype=ext_space["dyn/deter"].dtype
                ),
                "dyn/stoch": jnp.ones(
                    ext_space["dyn/stoch"].shape, dtype=ext_space["dyn/stoch"].dtype
                ),
                "action": jnp.ones(
                    actions_shape["action"].shape, dtype=actions_shape["action"].dtype
                ),
            }
            self.state = self.buffer.init(self.dummy_timestep)
            self.rng_key = jax.random.PRNGKey(buffer_seed)
            self._rng_lock = threading.Lock()

    def add(self, batch_sequence):
        with jax.default_device(self.cpu):
            self.state = self.buffer.add(self.state, batch_sequence)

    def sample(self):
        with self._rng_lock:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        with jax.default_device(self.cpu):
            batch, batch_indices, time_indices = self._sample_with_indices(self.state, rng_key)
        return batch, batch_indices, time_indices

    def update(self, batch_indices, time_indices, replay_updates):
        with jax.default_device(self.cpu):
            self.state = self._update_latents(
                self.state, batch_indices, time_indices, replay_updates
            )

    def can_sample(self):
        return self.buffer.can_sample(self.state)


class BufferPrioritised:
    def __init__(
        self,
        buffer_seed,
        add_batch_size,
        sample_batch_size,
        sample_sequence_length,
        period,
        min_length,
        max_size,
        priority_exponent,
        observation_shape,
        time_step_dtypes,
    ):
        self.dtypes = time_step_dtypes()
        self.cpu = jax.devices("cpu")[0]
        with jax.default_device(self.cpu):
            self.buffer = fbx.make_prioritised_trajectory_buffer(
                add_batch_size=add_batch_size,
                sample_batch_size=sample_batch_size,
                sample_sequence_length=sample_sequence_length,
                period=period,
                min_length_time_axis=min_length,
                max_length_time_axis=max_size // add_batch_size,
                priority_exponent=priority_exponent,
                device="cpu",
            )
            self.buffer = self.buffer.replace(
                init=jax.jit(self.buffer.init),
                add=jax.jit(self.buffer.add, donate_argnums=0),
                sample=jax.jit(self.buffer.sample),
                can_sample=jax.jit(self.buffer.can_sample),
                set_priorities=jax.jit(self.buffer.set_priorities, donate_argnums=0),
            )
            fake_timestep = TimeStepDQN(
                obs=jnp.ones(observation_shape, dtype=self.dtypes.obs),
                action=jnp.ones((), dtype=self.dtypes.action),
                reward=jnp.ones((), dtype=self.dtypes.reward),
                terminated=jnp.ones((), dtype=self.dtypes.terminated),
                truncated=jnp.ones((), dtype=self.dtypes.truncated),
            )
            self.state = self.buffer.init(fake_timestep)
            self.rng_key = jax.random.PRNGKey(buffer_seed)
            self._rng_lock = threading.Lock()

    def add(self, batch_sequence):
        with jax.default_device(self.cpu):
            self.state = self.buffer.add(self.state, batch_sequence)

    def sample(self):
        with self._rng_lock:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        with jax.default_device(self.cpu):
            batch = self.buffer.sample(self.state, rng_key)
        return batch

    def can_sample(self):
        return self.buffer.can_sample(self.state)

    def set_priorities(self, sample_indices, updated_priorities):
        self.state = self.buffer.set_priorities(
            self.state,
            jax.device_put(sample_indices, device=self.cpu),
            jax.device_put(updated_priorities, device=self.cpu),
        )
