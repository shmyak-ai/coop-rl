# Copyright 2026 The Coop RL Authors.
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

import math

import jax.numpy as jnp
import ml_collections
import numpy as np
from ml_collections import config_dict

from coop_rl.agents.mdqn import get_update_epoch
from coop_rl.agents.miqn import (
    get_select_action_recurrent_batch_fn,
    get_update_step_recurrent,
    restore_recurrent_dqn_flax_state,
)
from coop_rl.base.base_types import TimeStepDQNRecurrentDtypesAtari
from coop_rl.base.buffers import BufferTrajectoryDQNRecurrent
from coop_rl.base.environment import HandlerEnvAtari
from coop_rl.base.utils import make_optimizer
from coop_rl.networks.base import QuantileRecurrentNetwork
from coop_rl.networks.inputs import EmbeddingInput
from coop_rl.networks.quantile import NoisyDuelingQuantileQNetworkHead
from coop_rl.networks.resnet import DownsamplingStrategy, VisualResNetTorso
from coop_rl.networks.torso import DeepResidualTorso
from coop_rl.workers.auxiliary import Controller
from coop_rl.workers.collectors import CollectorDQNRecurrentUniform
from coop_rl.workers.trainers import Trainer


def get_config():
    config = ml_collections.ConfigDict()

    log_level = config_dict.FieldReference("INFO", field_type=str)
    observation_shape = config_dict.FieldReference(None, field_type=tuple)
    observation_dtype = config_dict.FieldReference(None, field_type=np.dtype)
    actions_shape = config_dict.FieldReference(None, field_type=np.integer)
    workdir = config_dict.FieldReference(None, field_type=str)
    checkpointdir = config_dict.FieldReference(None, field_type=str)

    seed = 73
    buffer_seed, trainer_seed, collectors_seed = seed + 1, seed + 2, seed + 3
    # Not an actual stopping condition here (the async Trainer stops at `steps` updates,
    # not env frames) — used to compute the epsilon schedule boundary and the update
    # budget below, the same way BTR's own env_frames budget does.
    env_frames = 32_000_000
    # miqn_btr_atari.py's env_frames * replay_ratio (1/64) = 500,000 gradient updates;
    # the async Trainer has no replay ratio, so match its update budget directly.
    steps = env_frames // 64
    training_iterations_per_step = 1

    hidden_state_dim = 512
    cell_type = "gru"
    burn_in_length = 10
    learn_length = 20
    sample_sequence_length = burn_in_length + learn_length
    batch_size = 32

    config.log_level = log_level
    config.num_collectors = 1
    config.num_samplers = 1
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.actions_shape = actions_shape
    config.workdir = workdir

    config.network = network = QuantileRecurrentNetwork
    config.args_network = args_network = ml_collections.ConfigDict()
    config.args_network.pre_torso = VisualResNetTorso
    config.args_network.args_pre_torso = ml_collections.ConfigDict()
    config.args_network.args_pre_torso.channels_per_group = (32, 64, 64)  # BTR: Impala width x2
    config.args_network.args_pre_torso.blocks_per_group = (2, 2, 2)
    config.args_network.args_pre_torso.downsampling_strategies = (
        DownsamplingStrategy.CONV_MAX,
    ) * 3
    config.args_network.args_pre_torso.hidden_sizes = (512,)  # Dense bridge into the RNN
    config.args_network.args_pre_torso.use_layer_norm = True  # BTR normalization (no spectral norm)
    config.args_network.args_pre_torso.activation = "relu"
    config.args_network.args_pre_torso.channel_first = False
    config.args_network.args_pre_torso.adaptive_pool_size = 6  # BTR adaptive maxpool, 11x11 -> 6x6
    config.args_network.args_pre_torso.dtype = jnp.bfloat16
    config.args_network.post_torso = DeepResidualTorso
    config.args_network.args_post_torso = ml_collections.ConfigDict()
    config.args_network.args_post_torso.width = 256
    config.args_network.args_post_torso.depth = 8
    config.args_network.args_post_torso.activation = "swish"
    config.args_network.args_post_torso.dtype = jnp.bfloat16
    config.args_network.head = NoisyDuelingQuantileQNetworkHead
    config.args_network.args_head = ml_collections.ConfigDict()
    config.args_network.args_head.action_dim = actions_shape
    config.args_network.args_head.epsilon = 0.01  # fallback only; schedule overrides below
    config.args_network.args_head.layer_sizes = [256]
    config.args_network.args_head.sigma_zero = 0.5  # Fortunato et al. (2017) default
    config.args_network.args_head.activation = "swish"
    config.args_network.args_head.n_cos = 64
    config.args_network.args_head.use_layer_norm = False
    config.args_network.args_head.dtype = jnp.bfloat16
    config.args_network.hidden_state_dim = hidden_state_dim
    config.args_network.cell_type = cell_type
    config.args_network.action_dim = actions_shape
    config.args_network.input_layer = EmbeddingInput

    config.optimizer = optimizer = make_optimizer
    config.args_optimizer = args_optimizer = ml_collections.ConfigDict()
    config.args_optimizer.init_lr = 1e-4  # BTR
    config.args_optimizer.decay_learning_rates = False
    config.args_optimizer.max_grad_norm = 10.0  # BTR
    # BTR: 0.005 / batch, with batch = transitions per update; here that is
    # sample_batch_size * learn_length = 16 * 20 = 320 (the heuristic's round number —
    # strictly only learn_length - n_steps anchors per sequence contribute).
    config.args_optimizer.adam_eps = 0.005 / (batch_size * learn_length)

    config.env = env = HandlerEnvAtari
    config.args_env = args_env = ml_collections.ConfigDict()
    config.args_env.env_name = "ale_py:ALE/Breakout-v5"
    config.args_env.stack_size = 1  # >= 1, 1 - no stacking
    config.args_env.num_envs = 8

    config.buffer = buffer = BufferTrajectoryDQNRecurrent
    config.args_buffer = args_buffer = ml_collections.ConfigDict()
    config.args_buffer.buffer_seed = buffer_seed
    config.args_buffer.add_batch_size = config.args_env.num_envs
    config.args_buffer.sample_batch_size = batch_size
    config.args_buffer.sample_sequence_length = sample_sequence_length
    config.args_buffer.period = learn_length // 2  # MEME-style 50%-overlapping learn windows
    # BTR min_sampling_size: 200,000 transitions before the first gradient update
    # (flashbax min_length is per time axis: 6250 * 32 add rows = 200k transitions).
    config.args_buffer.min_length = 6250
    config.args_buffer.max_size = 1000000  # in transitions
    config.args_buffer.observation_shape = observation_shape
    config.args_buffer.hidden_state_shape = (hidden_state_dim,)
    config.args_buffer.time_step_dtypes = time_step_dtypes = TimeStepDQNRecurrentDtypesAtari

    config.state_recover = state_recover = restore_recurrent_dqn_flax_state
    config.args_state_recover = args_state_recover = ml_collections.ConfigDict()
    config.args_state_recover.rng = None
    config.args_state_recover.network = network
    config.args_state_recover.args_network = args_network
    config.args_state_recover.optimizer = optimizer
    config.args_state_recover.args_optimizer = args_optimizer
    config.args_state_recover.observation_shape = observation_shape
    config.args_state_recover.hidden_state_dim = hidden_state_dim
    config.args_state_recover.cell_type = cell_type
    # tau is unused: target_update_period > 0 selects BTR's hard-copy path instead
    # of Polyak blending (see TrainState.apply_gradients in agents/miqn.py).
    config.args_state_recover.tau = 0.005
    config.args_state_recover.target_update_period = 500  # BTR: hard target copy
    config.args_state_recover.checkpointdir = checkpointdir

    config.controller = Controller
    config.args_controller = ml_collections.ConfigDict()
    config.args_controller.log_level = log_level

    config.trainer = Trainer
    config.args_trainer = ml_collections.ConfigDict()
    config.args_trainer.controller = None
    config.args_trainer.trainer_seed = trainer_seed
    config.args_trainer.log_level = log_level
    config.args_trainer.workdir = workdir
    config.args_trainer.steps = steps
    config.args_trainer.training_iterations_per_step = training_iterations_per_step
    config.args_trainer.summary_writing_period = 1000  # logging and reporting
    config.args_trainer.save_period = 10000  # orbax checkpointing
    config.args_trainer.synchronization_period = 10  # send params to control actor
    config.args_trainer.state_recover = state_recover
    config.args_trainer.args_state_recover = args_state_recover
    config.args_trainer.get_update_step = get_update_step_recurrent
    config.args_trainer.args_get_update_step = ml_collections.ConfigDict()
    config.args_trainer.args_get_update_step.apply_fn = None
    config.args_trainer.args_get_update_step.burn_in_length = burn_in_length
    config.args_trainer.args_get_update_step.n_steps = 3  # n-step TD targets, < learn_length
    config.args_trainer.args_get_update_step.gamma = 0.997  # BTR
    config.args_trainer.args_get_update_step.entropy_temperature = 0.03
    config.args_trainer.args_get_update_step.munchausen_coefficient = 0.9
    config.args_trainer.args_get_update_step.clip_value_min = -1.0
    config.args_trainer.args_get_update_step.quantile_huber_kappa = 1.0
    config.args_trainer.args_get_update_step.num_tau_samples = 64  # M-IQN paper
    config.args_trainer.args_get_update_step.num_tau_prime_samples = 64  # M-IQN paper
    config.args_trainer.args_get_update_step.max_abs_reward = 1.0
    config.args_trainer.args_get_update_step.obs_preprocess_fn = lambda x: (
        x.astype(jnp.bfloat16) / jnp.bfloat16(255.0)
    )
    config.args_trainer.args_get_update_step.recurrent_rollout_fn = None
    config.args_trainer.get_update_epoch = get_update_epoch
    config.args_trainer.args_get_update_epoch = ml_collections.ConfigDict()
    config.args_trainer.args_get_update_epoch.update_step_fn = None
    config.args_trainer.buffer = buffer
    config.args_trainer.args_buffer = args_buffer
    config.args_trainer.num_samples_on_gpu_cache = 3

    config.collector = CollectorDQNRecurrentUniform
    config.args_collector = ml_collections.ConfigDict()
    config.args_collector.controller = None
    config.args_collector.trainer = None
    config.args_collector.workdir = workdir
    config.args_collector.collectors_seed = collectors_seed
    config.args_collector.log_level = log_level
    config.args_collector.report_period = 10  # per rollouts sampled
    config.args_collector.steps_per_rollout = 200
    config.args_collector.state_recover = state_recover
    config.args_collector.args_state_recover = args_state_recover
    config.args_collector.env = env
    config.args_collector.args_env = args_env
    config.args_collector.get_select_action_fn = get_select_action_recurrent_batch_fn
    config.args_collector.args_get_select_action_fn = ml_collections.ConfigDict()
    config.args_collector.args_get_select_action_fn.apply_fn = None
    config.args_collector.args_get_select_action_fn.num_quantile_samples = 32  # M-IQN paper
    config.args_collector.args_get_select_action_fn.max_abs_reward = 1.0

    # BTR combines NoisyNets with annealed eps-greedy for the first half of training,
    # then relies on noisy weights alone. BTR's EpsilonGreedy subtracts
    # (eps - 0.01) / 2e6 per transition (eps_steps = 2M), i.e. an exponential-gap
    # decay eps(k) = 0.01 + 0.99 * exp(-k / 2e6): ~0.37 at 2M, ~0.14 at 4M, ~0.03 at
    # 8M frames. Disabled (eps = 0.0) from env_frames // 2 on, as in BTR. Each of this
    # async config's `num_collectors` collectors anneals independently over its own
    # local env-frame count (there is no single global frame counter), using the same
    # absolute boundaries as BTR's own schedule.
    def _btr_epsilon_schedule(step_count: int) -> float:
        if step_count >= env_frames // 2:
            return 0.0
        return 0.01 + 0.99 * math.exp(-step_count / 2_000_000)

    config.args_collector.args_get_select_action_fn.epsilon_scheduler_fn = _btr_epsilon_schedule
    config.args_collector.args_get_select_action_fn.obs_preprocess_fn = lambda x: (
        x.astype(jnp.float32) / 255.0
    )
    config.args_collector.hidden_state_dim = hidden_state_dim
    config.args_collector.cell_type = cell_type
    config.args_collector.time_step_dtypes = time_step_dtypes

    return config.lock()
