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

from coop_rl.replay_memory import circular_replay_buffer


@ray.remote(num_cpus=0)
class ControlActor:

    def __init__(self, num_collectors):
        self.num_collectors = num_collectors
        self.collector_id = 0
        self.done = False
        self.weights = None

    def set_done(self, done):
        self.done = done

    def is_done(self):
        return self.done

    def increment_collector_id(self):
        self.collector_id += 1
        if self.collector_id >= self.num_collectors:
            self.collector_id = 0

    def get_collector_id(self):
        return self.collector_id

    def set_weights(self, w):
        self.weights = w

    def get_weights(self):
        return self.weights


@ray.remote(num_cpus=1)
class ReplayActor:

    def __init__(self, config):
        self.buffer = circular_replay_buffer.OutOfGraphReplayBuffer(**config)

    def add_episode(self, episode_transitions):
        for observation, action, reward, terminal, *args, kwargs in episode_transitions:
            self.buffer.add(observation, action, reward, terminal, *args, **kwargs)
