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

import collections

import numpy as np
import reverb
import tensorflow as tf

from coop_rl.replay_memory import circular_replay_buffer


class ControlActor:
    def __init__(self):
        self.done = False
        self.parameters = None

    def set_done(self):
        self.done = True

    def is_done(self) -> bool:
        return self.done

    def set_parameters(self, w):
        self.parameters = w

    def get_parameters(self):
        return self.parameters

    def get_parameters_done(self):
        return self.parameters, self.done


class ReplayActorDopamine:
    def __init__(self, stack_size, **kwargs):
        if stack_size > 1:
            self.stacking = True
        self._replay = circular_replay_buffer.OutOfGraphReplayBuffer(stack_size=stack_size, **kwargs)

    def add_episode(self, episode_transitions):
        """
        Collectors use this method to add full episodes.
        """
        for observation, action, reward, terminated, *args, kwargs in episode_transitions:
            self._replay.add(observation, action, reward, terminated, *args, **kwargs)

    def add_count(self):
        """
        Returns:
            The number of transition additions to a replay buffer.
        """
        return self._replay.add_count

    def sample_from_replay_buffer(self):
        samples = self._replay.sample_transition_batch()
        types = self._replay.get_transition_elements()
        replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types, strict=False):
            if element_type.name in ("state", "next_state"):
                element = np.moveaxis(element, -1, 1) if self.stacking else element[..., 0]
            replay_elements[element_type.name] = element

        return replay_elements


class DQNUniformReverbServer:
    def __init__(
        self,
        batch_size,
        replay_capacity,
        observation_shape,
        timesteps,
        buffer_server_port=None,
        checkpointer=None
        ):

        min_size = batch_size
        max_size = replay_capacity

        observation_spec = tf.TensorSpec(observation_shape, tf.float32)
        action_spec = tf.TensorSpec([], tf.int32)
        rewards_spec = tf.TensorSpec([], tf.float32)
        dones_spec = tf.TensorSpec([], tf.float32)

        self._table_name = f"DQN_{timesteps}_timesteps_update"
        self._min_size = min_size
        self._server = reverb.Server(
            tables=[
                reverb.Table(
                    name=self._table_name,
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(max_size),
                    rate_limiter=reverb.rate_limiters.MinSize(min_size),
                    signature={
                        'observation':
                            tf.TensorSpec([timesteps, *observation_spec.shape], observation_spec.dtype),
                        'action':
                            tf.TensorSpec([timesteps, *action_spec.shape], action_spec.dtype),
                        'reward':
                            tf.TensorSpec([timesteps, *rewards_spec.shape], rewards_spec.dtype),
                        'terminated':
                            tf.TensorSpec([timesteps, *dones_spec.shape], dones_spec.dtype),
                    }
                )
            ],
            port=buffer_server_port,
            checkpointer=checkpointer
        )

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def server_port(self) -> int:
        return self._server.port
