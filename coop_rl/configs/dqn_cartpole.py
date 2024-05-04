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
from coop_rl.utils import identity_epsilon


def get_config():
    config = ml_collections.ConfigDict()
    config.replay = ml_collections.ConfigDict()
    config.collector = ml_collections.ConfigDict()
    config.agent = ml_collections.ConfigDict()
    config.agent.optimizer = ml_collections.ConfigDict()

    observation_shape = config_dict.FieldReference(None, field_type=tuple)
    observation_dtype = config_dict.FieldReference(None, field_type=np.dtype)
    num_actions = config_dict.FieldReference(None, field_type=np.integer)
    seed = config_dict.FieldReference(42)
    stack_size = config_dict.FieldReference(1)
    gamma = config_dict.FieldReference(0.99)
    environment = config_dict.FieldReference("CartPole-v1")
    network = config_dict.FieldReference(networks.ClassicControlDQNNetwork)

    config.debug = True
    config.seed = seed
    config.batch_size = 128
    config.num_collectors = 1
    config.environment = environment
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.num_actions = num_actions

    config.replay.replay_capacity = 100000
    config.replay.gamma = gamma
    config.replay.batch_size = 10
    config.replay.update_horizon = 1
    config.replay.stack_size = stack_size
    config.replay.observation_shape = observation_shape
    config.replay.observation_dtype = observation_dtype

    config.collector.num_actions = num_actions
    config.collector.observation_shape = observation_shape
    config.collector.observation_dtype = observation_dtype
    config.collector.stack_size = stack_size
    config.collector.environment = environment
    config.collector.network = network
    config.collector.seed = seed
    config.collector.epsilon_fn = identity_epsilon

    config.agent.loss_type = "huber"
    config.agent.gamma = gamma
    config.agent.update_period = 4
    config.agent.target_update_period = 100
    config.agent.num_actions = num_actions
    config.agent.observation_shape = observation_shape
    config.agent.observation_dtype = observation_dtype
    config.agent.stack_size = stack_size
    config.agent.network = network
    config.agent.seed = seed

    config.agent.optimizer.name = "adam"
    config.agent.optimizer.learning_rate = 0.001
    config.agent.optimizer.eps = 3.125e-4

    return config
