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

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    terminated: chex.Array
    truncated: chex.Array


class BufferFlat:
    def __init__(
        self, buffer_seed, max_length, min_length, sample_batch_size, add_sequences, add_batch_size, observation_shape
    ):
        self.buffer = fbx.make_flat_buffer(max_length, min_length, sample_batch_size, add_sequences, add_batch_size)
        fake_timestep = {"obs": jnp.array(observation_shape), "reward": jnp.array(1.0)}
        self.state = self.buffer.init(fake_timestep)
        self.rng_key = jax.random.PRNGKey(buffer_seed)

    def add(self, traj_batch):
        self.state = self.buffer.add(self.state, traj_batch)

    def sample(self):
        self.rng_key, rng_key = jax.random.split(self.rng_key)
        batch = self.buffer.sample(self.state, rng_key)
        return batch


class BufferTrajectory:
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
        self.dtypes = time_step_dtypes
        self.cpu = jax.devices("cpu")[0]
        with jax.default_device(self.cpu):
            self.buffer = fbx.make_trajectory_buffer(
                add_batch_size=add_batch_size,
                sample_batch_size=sample_batch_size,
                sample_sequence_length=sample_sequence_length,
                period=period,
                min_length_time_axis=min_length,
                max_size=max_size,
            )
            self.buffer = self.buffer.replace(
                init=jax.jit(self.buffer.init),
                add=jax.jit(self.buffer.add, donate_argnums=0),
                sample=jax.jit(self.buffer.sample),
                can_sample=jax.jit(self.buffer.can_sample),
            )
            fake_timestep = TimeStep(
                obs=jnp.ones(observation_shape, dtype=self.dtypes.obs),
                action=jnp.ones((), dtype=self.dtypes.action),
                reward=jnp.ones((), dtype=self.dtypes.reward),
                terminated=jnp.ones((), dtype=self.dtypes.terminated),
                truncated=jnp.ones((), dtype=self.dtypes.truncated),
            )
            self.state = self.buffer.init(fake_timestep)
            self.rng_key = jax.random.PRNGKey(buffer_seed)

    def add(self, traj_obs, traj_actions, traj_rewards, traj_terminated, traj_truncated):
        with jax.default_device(self.cpu):
            traj_batch_seq = TimeStep(
                obs=jnp.array(traj_obs, dtype=self.dtypes.obs),
                action=jnp.array(traj_actions, dtype=self.dtypes.action),
                reward=jnp.array(traj_rewards, dtype=self.dtypes.reward),
                terminated=jnp.array(traj_terminated, dtype=self.dtypes.terminated),
                truncated=jnp.array(traj_truncated, dtype=self.dtypes.truncated),
            )
            self.state = self.buffer.add(self.state, traj_batch_seq)

    def sample(self):
        self.rng_key, rng_key = jax.random.split(self.rng_key)
        with jax.default_device(self.cpu):
            batch = self.buffer.sample(self.state, rng_key)
        return batch

    def can_sample(self):
        return self.buffer.can_sample(self.state)
