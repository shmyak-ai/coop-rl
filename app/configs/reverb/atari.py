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
from coop_rl.utils import (
    HandlerEnvAtari,
    HandlerReverbReplay,
    HandlerReverbSampler,
    linearly_decaying_epsilon,
)
from coop_rl.workers import actors
from coop_rl.workers.collectors import DQNCollectorUniform


def get_config():
    config = ml_collections.ConfigDict()

    observation_shape = config_dict.FieldReference(None, field_type=tuple)
    observation_dtype = config_dict.FieldReference(None, field_type=np.dtype)
    num_actions = config_dict.FieldReference(None, field_type=np.integer)
    workdir = config_dict.FieldReference(None, field_type=str)
    table_name = config_dict.FieldReference(None, field_type=str)

    seed = 42
    num_collectors = 2
    replay_capacity = 1000000  # in transitions
    gamma = 0.99
    batch_size = 300  # > 1: target_q in dqn limitation
    stack_size = 4  # >= 1, 1 - no stacking
    timesteps = 4  # DQN n-steps update
    buffer_server_port = 8023
    env_name = "ALE/Breakout-v5"
    network = networks.NatureDQNNetwork

    config.seed = seed
    config.num_collectors = num_collectors
    config.env_name = env_name
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.num_actions = num_actions
    config.stack_size = stack_size
    config.workdir = workdir
    config.table_name = table_name

    config.control_actor = actors.ControlActor

    config.reverb_server = actors.DQNUniformReverbServer
    config.args_reverb_server = ml_collections.ConfigDict()
    config.args_reverb_server.batch_size = batch_size
    config.args_reverb_server.replay_capacity = replay_capacity
    config.args_reverb_server.observation_shape = observation_shape
    config.args_reverb_server.timesteps = timesteps
    config.args_reverb_server.buffer_server_port = buffer_server_port

    config.collector = DQNCollectorUniform
    config.args_collector = ml_collections.ConfigDict()
    config.args_collector.report_period = 100  # per episodes sampled
    config.args_collector.num_actions = num_actions
    config.args_collector.observation_shape = observation_shape
    config.args_collector.network = network
    config.args_collector.seed = seed
    config.args_collector.warmup_steps = 10000
    config.args_collector.epsilon_fn = linearly_decaying_epsilon
    config.args_collector.epsilon = 0.01
    config.args_collector.epsilon_decay_period = int(replay_capacity / 4 / num_collectors)
    config.args_collector.handler_env = HandlerEnvAtari
    config.args_collector.args_handler_env = ml_collections.ConfigDict()
    config.args_collector.args_handler_env.env_name = env_name
    config.args_collector.args_handler_env.stack_size = stack_size
    config.args_collector.handler_replay = HandlerReverbReplay
    config.args_collector.args_handler_replay = ml_collections.ConfigDict()
    config.args_collector.args_handler_replay.timesteps = timesteps
    config.args_collector.args_handler_replay.table_name = table_name
    config.args_collector.args_handler_replay.buffer_server_port = buffer_server_port

    config.agent = JaxDQNAgent
    config.args_agent = ml_collections.ConfigDict()
    config.args_agent.min_replay_history = 20000  # in transitions
    config.args_agent.training_steps = 1000000
    config.args_agent.workdir = workdir
    config.args_agent.loss_type = "huber"
    config.args_agent.gamma = gamma
    config.args_agent.batch_size = batch_size
    config.args_agent.update_horizon = timesteps - 1
    config.args_agent.target_update_period = 2000  # periods are in training_steps
    config.args_agent.summary_writing_period = 2000  # logging and reporting
    config.args_agent.save_period = 10000  # orbax checkpointing
    config.args_agent.observation_shape = observation_shape
    config.args_agent.seed = seed
    config.args_agent.network = network
    config.args_agent.args_network = ml_collections.ConfigDict()
    config.args_agent.args_network.num_actions = num_actions
    config.args_agent.optimizer = optax.adam
    config.args_agent.args_optimizer = ml_collections.ConfigDict()
    config.args_agent.args_optimizer.learning_rate = 6.25e-5
    config.args_agent.args_optimizer.eps = 1.5e-4
    config.args_agent.handler_sampler = HandlerReverbSampler
    config.args_agent.args_handler_sampler = ml_collections.ConfigDict()
    config.args_agent.args_handler_sampler.gamma = gamma
    config.args_agent.args_handler_sampler.batch_size = batch_size
    config.args_agent.args_handler_sampler.timesteps = timesteps
    config.args_agent.args_handler_sampler.table_name = table_name
    config.args_agent.args_handler_sampler.buffer_server_port = buffer_server_port

    return config
