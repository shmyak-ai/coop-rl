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

from coop_rl.agents.dqn import get_update_epoch, get_update_step, restore_dqn_flax_state
from coop_rl.base_types import ClassicControlTimeStepDtypes
from coop_rl.buffers import BufferTrajectory
from coop_rl.environment import HandlerEnv
from coop_rl.networks.base import FeedForwardActor, get_actor
from coop_rl.networks.heads import DiscreteQNetworkHead
from coop_rl.networks.inputs import EmbeddingInput
from coop_rl.networks.torso import MLPTorso
from coop_rl.workers.auxiliary import Controller
from coop_rl.workers.collectors import DQNCollectorUniform
from coop_rl.workers.trainers import Trainer


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
    config.num_collectors = num_collectors = 5
    config.num_samplers = 5
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.num_actions = num_actions
    config.workdir = workdir

    config.network = network = get_actor
    config.args_network = args_network = ml_collections.ConfigDict()
    config.args_network.base = FeedForwardActor
    config.args_network.torso = MLPTorso
    config.args_network.args_torso = ml_collections.ConfigDict()
    config.args_network.args_torso.activation = 'silu'
    config.args_network.args_torso.layer_sizes = [256, 256]
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

    config.env = env = HandlerEnv
    config.args_env = args_env = ml_collections.ConfigDict()
    config.args_env.env_name = "CartPole-v1"
    config.args_env.stack_size = 1  # >= 1, 1 - no stacking

    config.buffer = buffer = BufferTrajectory
    config.args_buffer = args_buffer = ml_collections.ConfigDict()
    config.args_buffer.buffer_seed = buffer_seed
    config.args_buffer.add_batch_size = num_collectors
    config.args_buffer.sample_batch_size = 100
    config.args_buffer.sample_sequence_length = 5  # DQN n-steps update
    config.args_buffer.period = 1
    config.args_buffer.min_length = 1000
    config.args_buffer.max_size = 100000  # in transitions
    config.args_buffer.observation_shape = observation_shape
    config.args_buffer.time_step_dtypes = time_step_dtypes = ClassicControlTimeStepDtypes()

    config.agent_params = agent_params = ml_collections.ConfigDict()
    config.agent_params.tau = tau = 0.005  # smoothing coefficient for target networks
    config.agent_params.gamma = 0.99  # discount factor
    config.agent_params.huber_loss_parameter = 0.0  # parameter for the huber loss. If 0, it uses MSE loss
    config.agent_params.max_abs_reward = 1000.0

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

    config.trainer = Trainer
    config.args_trainer = ml_collections.ConfigDict()
    config.args_trainer.trainer_seed = trainer_seed
    config.args_trainer.log_level = log_level
    config.args_trainer.workdir = workdir
    config.args_trainer.steps = 100000
    config.args_trainer.training_iterations_per_step = 1  # to increase gpu load
    config.args_trainer.summary_writing_period = 1000  # logging and reporting
    config.args_trainer.save_period = 10000  # orbax checkpointing
    config.args_trainer.synchronization_period = 100  # send params to control actor
    config.args_trainer.state_recover = state_recover
    config.args_trainer.args_state_recover = args_state_recover
    config.args_trainer.get_update_step = get_update_step
    config.args_trainer.get_update_epoch = get_update_epoch
    config.args_trainer.agent_params = agent_params
    config.args_trainer.buffer = buffer
    config.args_trainer.args_buffer = args_buffer
    config.args_trainer.num_samples_on_gpu_cache = 75 
    config.args_trainer.num_samples_to_gpu = 15

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
    config.args_collector.time_step_dtypes = time_step_dtypes

    return config
