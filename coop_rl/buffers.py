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

from coop_rl.agents.dreamer import Agent
from coop_rl.base_types import TimeStep, TimeStepDreamer


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


class BufferTrajectoryDreamer:
    def __init__(
        self,
        dreamer_config,
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
        ext_space = Agent(observation_shape, actions_shape, dreamer_config).ext_space
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
            breakpoint()
            self.dummy_timestep = TimeStepDreamer(
                image=jnp.ones(observation_shape["image"].shape, dtype=observation_shape["image"].dtype),
                is_first=jnp.ones(observation_shape["is_first"].shape, dtype=observation_shape["is_first"].dtype),
                is_last=jnp.ones(observation_shape["is_last"].shape, dtype=observation_shape["is_last"].dtype),
                is_terminal=jnp.ones(
                    observation_shape["is_terminal"].shape, dtype=observation_shape["is_terminal"].dtype
                ),
                reward=jnp.ones(observation_shape["reward"].shape, dtype=observation_shape["reward"].dtype),
                consec=jnp.ones(ext_space["consec"].shape, dtype=ext_space["consec"].dtype),
                stepid=jnp.ones(ext_space["stepid"].shape, dtype=ext_space["stepid"].dtype),
                dyn_deter=jnp.ones(ext_space["dyn/deter"].shape, dtype=ext_space["dyn/deter"].dtype),
                dyn_stoch=jnp.ones(ext_space["dyn/stoch"].shape, dtype=ext_space["dyn/stoch"].dtype),
                action=jnp.ones(actions_shape["action"].shape, dtype=actions_shape["action"].dtype),
            )
            self.state = self.buffer.init(self.dummy_timestep)
            self.rng_key = jax.random.PRNGKey(buffer_seed)

    def add(self, traj_obs, traj_actions, traj_rewards, traj_terminated, traj_truncated):
        with jax.default_device(self.cpu):
            traj_batch_seq = TimeStepDreamer(
                image=jnp.array(traj_obs["image"], dtype=self.dtypes.obs),
                reward=jnp.array(traj_obs["reward"], dtype=self.dtypes.obs),
                is_first=jnp.array(traj_obs["is_first"], dtype=self.dtypes.obs),
                is_last=jnp.array(traj_obs["is_last"], dtype=self.dtypes.obs),
                is_terminal=jnp.array(traj_obs["is_terminal"], dtype=self.dtypes.obs),
                action=jnp.array((), dtype=self.dtypes.action),
                dyn_deter=jnp.array((), dtype=self.dtypes.terminated),
                dyn_stoch=jnp.array((), dtype=self.dtypes.truncated),
            )
            self.state = self.buffer.add(self.state, traj_batch_seq)

    def sample(self):
        self.rng_key, rng_key = jax.random.split(self.rng_key)
        with jax.default_device(self.cpu):
            batch = self.buffer.sample(self.state, rng_key)
        return batch

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
        self.dtypes = time_step_dtypes
        self.cpu = jax.devices("cpu")[0]
        with jax.default_device(self.cpu):
            self.buffer = fbx.make_prioritised_trajectory_buffer(
                add_batch_size=add_batch_size,
                sample_batch_size=sample_batch_size,
                sample_sequence_length=sample_sequence_length,
                period=period,
                min_length_time_axis=min_length,
                max_size=max_size,
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
        with jax.default_device(self.cpu):
            self.rng_key, rng_key = jax.random.split(self.rng_key)
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
