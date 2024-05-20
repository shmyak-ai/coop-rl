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

import ray


@ray.remote(num_cpus=0)
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


@ray.remote(num_cpus=1)
class ReplayActor:

    def __init__(self, config):
        self._replay = config.replay(**config.args_replay)

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
        # dopamine replay buffer return 'states' which have the last 'stack' dimension 
        # in states and next_states, we don't use it so far
        for element, element_type in zip(samples, types, strict=False):
            if element_type.name in ('state', 'next_state'):
                element = element[..., 0]
            replay_elements[element_type.name] = element

        return replay_elements
