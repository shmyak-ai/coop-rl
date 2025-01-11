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

import ml_collections
import numpy as np
import optax
from ml_collections import config_dict

from coop_rl.agents.dqn import DQN, restore_dqn_flax_state
from coop_rl.buffers import BufferTrajectory
from coop_rl.environment import HandlerEnvAtari
from coop_rl.networks.base import FeedForwardActor, get_actor
from coop_rl.networks.heads import DiscreteQNetworkHead
from coop_rl.networks.inputs import EmbeddingInput
from coop_rl.networks.torso import CNNTorso
from coop_rl.workers.auxiliary import Controller
from coop_rl.workers.collectors import DQNCollectorUniform


def get_config():
    config = ml_collections.ConfigDict()

    log_level = config_dict.FieldReference("INFO", field_type=str)
    observation_shape = config_dict.FieldReference(None, field_type=tuple)
    observation_dtype = config_dict.FieldReference(None, field_type=np.dtype)
    num_actions = config_dict.FieldReference(None, field_type=np.integer)
    workdir = config_dict.FieldReference(None, field_type=str)
    checkpointdir = config_dict.FieldReference(None, field_type=str)

    seed = 73
    buffer_seed, trainer_seed, collectors_seed = seed + 1, seed + 2, seed + 3

    config.log_level = log_level
    config.num_collectors = num_collectors = 2
    config.num_samplers = 8
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.num_actions = num_actions
    config.workdir = workdir

    config.network = network = get_actor
    config.args_network = args_network = ml_collections.ConfigDict()
    config.args_network.base = FeedForwardActor
    config.args_network.torso = CNNTorso
    config.args_network.args_torso = ml_collections.ConfigDict()
    config.args_network.args_torso.activation = 'silu'
    config.args_network.args_torso.channel_first = False
    config.args_network.args_torso.channel_sizes = [32, 64, 64]
    config.args_network.args_torso.kernel_sizes = [8, 4, 3]
    config.args_network.args_torso.strides = [4, 2, 1]
    config.args_network.args_torso.hidden_sizes = [128, 128]
    config.args_network.args_torso.use_layer_norm = False
    config.args_network.action_head = DiscreteQNetworkHead
    config.args_network.args_action_head = ml_collections.ConfigDict()
    config.args_network.args_action_head.action_dim = num_actions
    config.args_network.args_action_head.epsilon = 0.1
    config.args_network.input_layer = EmbeddingInput

    config.optimizer = optimizer = optax.adam 
    config.args_optimizer = args_optimizer = ml_collections.ConfigDict()
    config.args_optimizer.learning_rate = 6.25e-5
    config.args_optimizer.eps = 1e-5

    config.env = env = HandlerEnvAtari
    config.args_env = args_env = ml_collections.ConfigDict()
    config.args_env.env_name = "ale_py:ALE/Breakout-v5"
    config.args_env.stack_size = 4  # >= 1, 1 - no stacking

    config.buffer = buffer = BufferTrajectory
    config.args_buffer = args_buffer = ml_collections.ConfigDict()
    config.args_buffer.buffer_seed = buffer_seed
    config.args_buffer.add_batch_size = num_collectors
    config.args_buffer.sample_batch_size = 32
    config.args_buffer.sample_sequence_length = 3  # DQN n-steps update
    config.args_buffer.period = 1
    config.args_buffer.min_length = 1000
    config.args_buffer.max_size = 100000  # in transitions
    config.args_buffer.observation_shape = observation_shape

    config.dqn_params = dqn_params = ml_collections.ConfigDict()
    config.dqn_params.tau = tau = 0.005  # smoothing coefficient for target networks
    config.dqn_params.gamma = 0.99  # discount factor
    config.dqn_params.huber_loss_parameter = 0.0  # parameter for the huber loss. If 0, it uses MSE loss
    config.dqn_params.max_abs_reward = 1000.0

    config.state_recover = state_recover = restore_dqn_flax_state
    config.args_state_recover = args_state_recover = ml_collections.ConfigDict()
    config.args_state_recover.network = network
    config.args_state_recover.args_network = args_network
    config.args_state_recover.optimizer = optimizer 
    config.args_state_recover.args_optimizer = args_optimizer 
    config.args_state_recover.observation_shape = observation_shape
    config.args_state_recover.tau = tau
    config.args_state_recover.checkpointdir = checkpointdir

    config.controller = Controller

    config.trainer = DQN
    config.args_trainer = ml_collections.ConfigDict()
    config.args_trainer.trainer_seed = trainer_seed
    config.args_trainer.log_level = log_level
    config.args_trainer.workdir = workdir
    config.args_trainer.steps = 10000000
    config.args_trainer.training_iterations_per_step = 1  # to increase gpu load ?
    config.args_trainer.summary_writing_period = 100  # logging and reporting
    config.args_trainer.save_period = 10000  # orbax checkpointing
    config.args_trainer.synchronization_period = 100  # send params to control actor
    config.args_trainer.observation_shape = observation_shape
    config.args_trainer.state_recover = state_recover
    config.args_trainer.args_state_recover = args_state_recover
    config.args_trainer.dqn_params = dqn_params
    config.args_trainer.buffer = buffer
    config.args_trainer.args_buffer = args_buffer
    config.args_trainer.network = network
    config.args_trainer.args_network = args_network
    config.args_trainer.optimizer = optimizer 
    config.args_trainer.args_optimizer = args_optimizer 
    config.args_trainer.num_samples_on_gpu_cache = 30 
    config.args_trainer.num_samples_to_gpu = 50

    config.collector = DQNCollectorUniform
    config.args_collector = ml_collections.ConfigDict()
    config.args_collector.collectors_seed = collectors_seed
    config.args_collector.log_level = log_level
    config.args_collector.report_period = 100  # per rollouts sampled
    config.args_collector.observation_shape = observation_shape
    config.args_collector.network = network
    config.args_collector.args_network = args_network
    config.args_collector.state_recover = state_recover
    config.args_collector.args_state_recover = args_state_recover
    config.args_collector.env = env
    config.args_collector.args_env = args_env

    return config
