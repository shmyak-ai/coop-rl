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

"""DQN with a Gym Cartpole environment Hyperparameter configuration."""

import ml_collections
import numpy as np
from ml_collections import config_dict

from coop_rl import networks
from coop_rl.replay_memory import circular_replay_buffer
from coop_rl.utils import identity_epsilon
from coop_rl.workers.collectors import DQNCollectorUniform


def get_config():
    config = ml_collections.ConfigDict()
    config.agent = ml_collections.ConfigDict()
    config.agent.optimizer = ml_collections.ConfigDict()

    observation_shape = config_dict.FieldReference(None, field_type=tuple)
    observation_dtype = config_dict.FieldReference(None, field_type=np.dtype)
    num_actions = config_dict.FieldReference(None, field_type=np.integer)
    seed = config_dict.FieldReference(42)
    gamma = config_dict.FieldReference(0.99)
    batch_size = config_dict.FieldReference(128)
    environment_name = config_dict.FieldReference("CartPole-v1")
    network = config_dict.FieldReference(networks.ClassicControlDQNNetwork)

    config.local = False
    config.seed = seed
    config.num_collectors = 3
    config.environment_name = environment_name
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.num_actions = num_actions

    config.replay = circular_replay_buffer.OutOfGraphReplayBuffer
    config.args_replay = ml_collections.ConfigDict()
    config.args_replay.replay_capacity = 100000
    config.args_replay.gamma = gamma
    config.args_replay.batch_size = batch_size
    config.args_replay.update_horizon = 1
    config.args_replay.observation_shape = observation_shape
    config.args_replay.observation_dtype = observation_dtype

    config.collector = DQNCollectorUniform
    config.args_collector = ml_collections.ConfigDict()
    config.args_collector.num_actions = num_actions
    config.args_collector.observation_shape = observation_shape
    config.args_collector.observation_dtype = observation_dtype
    config.args_collector.environment_name = environment_name
    config.args_collector.network = network
    config.args_collector.seed = seed
    config.args_collector.epsilon_fn = identity_epsilon

    config.agent.loss_type = "huber"
    config.agent.gamma = gamma
    config.agent.update_period = 4
    config.agent.target_update_period = 100
    config.agent.num_actions = num_actions
    config.agent.observation_shape = observation_shape
    config.agent.observation_dtype = observation_dtype
    config.agent.network = network
    config.agent.seed = seed

    config.agent.optimizer.name = "adam"
    config.agent.optimizer.learning_rate = 0.001
    config.agent.optimizer.eps = 3.125e-4

    return config
