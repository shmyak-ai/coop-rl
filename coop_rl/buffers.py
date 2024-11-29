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

import flashbax as fbx
import jax
import jax.numpy as jnp


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
        max_length,
        min_length,
        sample_batch_size,
        add_batch_size,
        sample_sequence_length,
        period,
        observation_shape,
    ):
        self.buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=max_length,
            min_length_time_axis=min_length,
            sample_batch_size=sample_batch_size,
            add_batch_size=add_batch_size,
            sample_sequence_length=sample_sequence_length,
            period=period,
        )
        fake_timestep = {"obs": jnp.array(observation_shape), "reward": jnp.array(1.0)}
        self.state = self.buffer.init(fake_timestep)
        self.rng_key = jax.random.PRNGKey(buffer_seed)

    def add(self, traj_batch_seq):
        self.state = self.buffer.add(self.state, traj_batch_seq)

    def sample(self):
        self.rng_key, rng_key = jax.random.split(self.rng_key)
        batch = self.buffer.sample(self.state, rng_key)
        return batch
