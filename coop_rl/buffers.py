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
        add_batch_size,
        sample_batch_size,
        sample_sequence_length,
        period,
        min_length,
        max_size,
        observation_shape,
    ):
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
            fake_timestep = { 
                "obs": jnp.ones(observation_shape, dtype="float32"),
                "action": jnp.array(1.0, dtype="float32"),
                "reward": jnp.array(1.0, dtype="float32"),
                "terminated": jnp.array(1.0, dtype="float32"),
                "truncated": jnp.array(1.0, dtype="float32"),
            }
            self.state = self.buffer.init(fake_timestep)
            self.rng_key = jax.random.PRNGKey(buffer_seed)

    def add(self, traj_obs, traj_actions, traj_rewards, traj_terminated, traj_truncated):
        with jax.default_device(self.cpu):
            traj_batch_seq = {
                "obs": jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32), traj_obs),
                "action": jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32), traj_actions),
                "reward": jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32), traj_rewards),
                "terminated": jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32), traj_terminated),
                "truncated": jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32), traj_truncated),
            }
            self.state = self.buffer.add(self.state, traj_batch_seq)

    def sample(self):
        self.rng_key, rng_key = jax.random.split(self.rng_key)
        with jax.default_device(self.cpu):
            batch = self.buffer.sample(self.state, rng_key)
        return batch

    def can_sample(self):
        return self.buffer.can_sample(self.state)
