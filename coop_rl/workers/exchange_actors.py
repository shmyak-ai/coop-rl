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
        self.buffer = config.replay(**config.args_replay)

    def add_episode(self, episode_transitions):
        """
        Collectors use this method to add full episodes.
        """
        for observation, action, reward, terminated, *args, kwargs in episode_transitions:
            self.buffer.add(observation, action, reward, terminated, *args, **kwargs)
    
    def add_count(self):
        """
        Returns:
            The number of transition additions to a replay buffer.
        """
        return self.buffer.add_count
