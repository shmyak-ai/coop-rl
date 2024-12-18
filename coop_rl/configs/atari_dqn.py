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

from coop_rl.agents.dqn import DQN, restore_dqn_flax_state
from coop_rl.buffers import BufferTrajectory
from coop_rl.environment import HandlerEnvAtari
from coop_rl.networks import NatureDQNNetwork
from coop_rl.utils import linearly_decaying_epsilon
from coop_rl.workers.auxiliary import Controller
from coop_rl.workers.collectors import DQNCollectorUniform


def get_config():
    config = ml_collections.ConfigDict()

    log_level = config_dict.FieldReference(None, field_type=str)
    observation_shape = config_dict.FieldReference(None, field_type=tuple)
    observation_dtype = config_dict.FieldReference(None, field_type=np.dtype)
    num_actions = config_dict.FieldReference(None, field_type=np.integer)
    workdir = config_dict.FieldReference(None, field_type=str)
    checkpointdir = config_dict.FieldReference(None, field_type=str)
    flax_state = config_dict.FieldReference(None, field_type=object)

    log_level = "INFO"
    seed = 42
    num_collectors = 2
    buffer_max_size = 100000  # in transitions
    learning_rate = 6.25e-5
    eps = 1.5e-4
    gamma = 0.99
    batch_size = 32  # > 1: target_q in dqn limitation
    stack_size = 4  # >= 1, 1 - no stacking
    timesteps = 5  # DQN n-steps update
    env_name = "ale_py:ALE/Breakout-v5"
    network = NatureDQNNetwork
    optimizer = optax.adam
    controller = Controller
    flax_state = None

    seed = 73
    buffer_seed, trainer_seed, collectors_seed = seed + 1, seed + 2, seed + 3

    config.log_level = log_level
    config.num_collectors = num_collectors
    config.env_name = env_name
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.num_actions = num_actions
    config.stack_size = stack_size
    config.workdir = workdir

    config.state_recover = restore_dqn_flax_state
    config.args_state_recover = ml_collections.ConfigDict()
    config.args_state_recover.num_actions = num_actions
    config.args_state_recover.network = network
    config.args_state_recover.optimizer = optimizer
    config.args_state_recover.observation_shape = observation_shape
    config.args_state_recover.learning_rate = learning_rate
    config.args_state_recover.eps = eps
    config.args_state_recover.checkpointdir = checkpointdir

    config.buffer = buffer = BufferTrajectory
    config.args_buffer = args_buffer = ml_collections.ConfigDict()
    config.args_buffer.buffer_seed = buffer_seed
    config.args_buffer.add_batch_size = num_collectors
    config.args_buffer.sample_batch_size = batch_size
    config.args_buffer.sample_sequence_length = timesteps
    config.args_buffer.period = 1
    config.args_buffer.min_length = 1000
    config.args_buffer.max_size = buffer_max_size
    config.args_buffer.observation_shape = observation_shape

    config.network = network
    config.args_network = args_network = ml_collections.ConfigDict()
    config.args_network.num_actions = num_actions

    config.optimizer = optimizer 
    config.args_optimizer = args_optimizer = ml_collections.ConfigDict()
    config.args_optimizer.learning_rate = learning_rate
    config.args_optimizer.eps = eps

    config.env = env = HandlerEnvAtari
    config.args_env = args_env = ml_collections.ConfigDict()
    config.args_env.env_name = env_name
    config.args_env.stack_size = stack_size

    config.controller = controller

    config.trainer = DQN
    config.args_trainer = ml_collections.ConfigDict()
    config.args_trainer.trainer_seed = trainer_seed
    config.args_trainer.log_level = log_level
    config.args_trainer.workdir = workdir
    config.args_trainer.training_steps = 1000000
    config.args_trainer.loss_type = "mse"
    config.args_trainer.gamma = gamma
    config.args_trainer.batch_size = batch_size
    config.args_trainer.update_horizon = timesteps - 1
    config.args_trainer.target_update_period = 2000  # periods are in training_steps
    config.args_trainer.summary_writing_period = 2000  # logging and reporting
    config.args_trainer.save_period = 30000  # orbax checkpointing
    config.args_trainer.synchronization_period = 10  # send params to control actor
    config.args_trainer.observation_shape = observation_shape
    config.args_trainer.flax_state = flax_state
    config.args_trainer.buffer = buffer
    config.args_trainer.args_buffer = args_buffer
    config.args_trainer.network = network
    config.args_trainer.args_network = args_network
    config.args_trainer.optimizer = optimizer 
    config.args_trainer.args_optimizer = args_optimizer 

    config.collector = DQNCollectorUniform
    config.args_collector = ml_collections.ConfigDict()
    config.args_collector.collectors_seed = collectors_seed
    config.args_collector.log_level = log_level
    config.args_collector.report_period = 25  # per episodes sampled
    config.args_collector.num_actions = num_actions
    config.args_collector.observation_shape = observation_shape
    config.args_collector.network = network
    config.args_collector.args_network = args_network
    config.args_collector.warmup_steps = 10000
    config.args_collector.epsilon_fn = linearly_decaying_epsilon
    config.args_collector.epsilon = 0.01
    config.args_collector.epsilon_decay_period = int(buffer_max_size / 4 / num_collectors)
    config.args_collector.flax_state = flax_state
    config.args_collector.env = env
    config.args_collector.args_env = args_env

    return config
