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

"""DQN with a Gym Cartpole environment configuration."""

import ml_collections
import numpy as np
import optax
from ml_collections import config_dict

from coop_rl import networks
from coop_rl.agents.dqn import JaxDQNAgent
from coop_rl.replay_memory import circular_replay_buffer
from coop_rl.utils import identity_epsilon
from coop_rl.workers.collectors import DQNCollectorUniform


def get_config():
    config = ml_collections.ConfigDict()

    observation_shape = config_dict.FieldReference(None, field_type=tuple)
    observation_dtype = config_dict.FieldReference(None, field_type=np.dtype)
    num_actions = config_dict.FieldReference(None, field_type=np.integer)
    workdir = config_dict.FieldReference(None, field_type=str)
    seed = config_dict.FieldReference(42)
    gamma = config_dict.FieldReference(0.99)
    batch_size = config_dict.FieldReference(1000)  # > 1: target_q in dqn limitation
    update_horizon = config_dict.FieldReference(3)
    environment_name = config_dict.FieldReference("CartPole-v1")
    network = config_dict.FieldReference(networks.ClassicControlDQNNetwork)

    config.seed = seed
    config.num_collectors = 3
    config.environment_name = environment_name
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.num_actions = num_actions
    config.workdir = workdir

    config.replay = circular_replay_buffer.OutOfGraphReplayBuffer
    config.args_replay = ml_collections.ConfigDict()
    config.args_replay.replay_capacity = 1000000  # in transitions
    config.args_replay.gamma = gamma
    config.args_replay.batch_size = batch_size
    config.args_replay.update_horizon = update_horizon
    config.args_replay.observation_shape = observation_shape
    config.args_replay.observation_dtype = observation_dtype

    config.collector = DQNCollectorUniform
    config.args_collector = ml_collections.ConfigDict()
    config.args_collector.num_actions = num_actions
    config.args_collector.observation_shape = observation_shape
    config.args_collector.environment_name = environment_name
    config.args_collector.network = network
    config.args_collector.seed = seed
    config.args_collector.epsilon_fn = identity_epsilon

    config.agent = JaxDQNAgent
    config.args_agent = ml_collections.ConfigDict()
    config.args_agent.min_replay_history = 10000  # in transitions
    config.args_agent.training_steps = 10000
    config.args_agent.num_actions = num_actions
    config.args_agent.workdir = workdir
    config.args_agent.loss_type = "huber"
    config.args_agent.gamma = gamma
    config.args_agent.batch_size = batch_size
    config.args_agent.update_horizon = update_horizon
    config.args_agent.target_update_period = 100  # periods are in training_steps
    config.args_agent.synchronization_period = 100  # send parameters to contol actor
    config.args_agent.summary_writing_period = 100  # tensorflow logging and reporting
    config.args_agent.observation_shape = observation_shape
    config.args_agent.network = network
    config.args_agent.seed = seed
    config.args_agent.optimizer = optax.adam
    config.args_agent.args_optimizer = ml_collections.ConfigDict()
    config.args_agent.args_optimizer.learning_rate = 0.001
    config.args_agent.args_optimizer.eps = 3.125e-4

    return config
