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

import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from ml_collections import config_dict

from coop_rl.agents.miqn import (
    get_select_action_batch_fn,
    get_update_epoch,
    get_update_step,
    restore_dqn_flax_state,
)
from coop_rl.base.base_types import TimeStepDQNDtypesAtari
from coop_rl.base.buffers import BufferPrioritised
from coop_rl.base.environment import HandlerEnvAtari
from coop_rl.base.utils import make_optimizer
from coop_rl.networks.base import QuantileFeedForwardNetwork
from coop_rl.networks.inputs import EmbeddingInput
from coop_rl.networks.quantile import NoisyDuelingQuantileQNetworkHead
from coop_rl.networks.resnet import DownsamplingStrategy, VisualResNetTorso
from coop_rl.workers.collectors import CollectorDQNUniform
from coop_rl.workers.trainers import TrainerSequential


def get_config():
    # M-IQN with the "Beyond The Rainbow" (arXiv:2411.03820) encoder and hyperparameters:
    # Impala ResNet (2x width) + LayerNorm + adaptive maxpool, gamma 0.997, PER alpha 0.2,
    # grad clip 10, 8 quantile samples. LayerNorm replaces BTR's spectral norm.
    # Uses TrainerSequential (run with --backend sequential): BTR's own loop does
    # exactly one gradient update per vectorized env-step batch, so this reproduces
    # that synchronous collect/train coupling instead of coop-rl's async Ray backend.
    config = ml_collections.ConfigDict()

    log_level = config_dict.FieldReference("INFO", field_type=str)
    observation_shape = config_dict.FieldReference(None, field_type=tuple)
    observation_dtype = config_dict.FieldReference(None, field_type=np.dtype)
    actions_shape = config_dict.FieldReference(None, field_type=np.integer)
    workdir = config_dict.FieldReference(None, field_type=str)
    checkpointdir = config_dict.FieldReference(None, field_type=str)

    seed = 73
    buffer_seed, trainer_seed, collectors_seed = seed + 1, seed + 2, seed + 3
    num_envs = 32
    replay_ratio = 1 / num_envs  # BTR: exactly one learn() call per env-step batch
    # Preserves the previous async config's 1,000,000-gradient-update budget:
    # env_frames * replay_ratio == 1,000,000 updates.
    env_frames = 32_000_000
    batch_size = 256

    config.log_level = log_level
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.actions_shape = actions_shape
    config.workdir = workdir

    config.network = network = QuantileFeedForwardNetwork
    config.args_network = args_network = ml_collections.ConfigDict()
    config.args_network.torso = VisualResNetTorso
    config.args_network.args_torso = ml_collections.ConfigDict()
    config.args_network.args_torso.channels_per_group = (32, 64, 64)  # BTR: Impala width x2
    config.args_network.args_torso.blocks_per_group = (2, 2, 2)
    config.args_network.args_torso.downsampling_strategies = (DownsamplingStrategy.CONV_MAX,) * 3
    config.args_network.args_torso.hidden_sizes = ()  # flatten only: 6*6*64 = 2304 IQN embedding
    config.args_network.args_torso.use_layer_norm = True  # BTR normalization (no spectral norm)
    config.args_network.args_torso.activation = "relu"
    config.args_network.args_torso.channel_first = False
    config.args_network.args_torso.adaptive_pool_size = 6  # BTR adaptive maxpool, 11x11 -> 6x6
    config.args_network.args_torso.dtype = jnp.bfloat16
    config.args_network.head = NoisyDuelingQuantileQNetworkHead
    config.args_network.args_head = ml_collections.ConfigDict()
    config.args_network.args_head.action_dim = actions_shape
    config.args_network.args_head.epsilon = 0.01  # fallback only; schedule overrides below
    config.args_network.args_head.layer_sizes = [512]  # BTR dueling streams
    config.args_network.args_head.sigma_zero = 0.5  # Fortunato et al. (2017) default
    config.args_network.args_head.activation = "relu"
    config.args_network.args_head.n_cos = 64
    config.args_network.args_head.use_layer_norm = False
    config.args_network.args_head.dtype = jnp.bfloat16
    config.args_network.input_layer = EmbeddingInput

    config.optimizer = optimizer = make_optimizer
    config.args_optimizer = args_optimizer = ml_collections.ConfigDict()
    config.args_optimizer.init_lr = 1e-4  # BTR (same batch size, so no reason to diverge)
    config.args_optimizer.decay_learning_rates = False
    config.args_optimizer.max_grad_norm = 10.0  # BTR
    config.args_optimizer.adam_eps = 0.005 / batch_size  # BTR: 0.005 / batch_size

    config.env = env = HandlerEnvAtari
    config.args_env = args_env = ml_collections.ConfigDict()
    config.args_env.env_name = "ale_py:ALE/Breakout-v5"
    config.args_env.stack_size = 4  # >= 1, 1 - no stacking
    config.args_env.num_envs = num_envs

    config.buffer = buffer = BufferPrioritised
    config.args_buffer = args_buffer = ml_collections.ConfigDict()
    config.args_buffer.buffer_seed = buffer_seed
    config.args_buffer.add_batch_size = config.args_env.num_envs
    # BTR batch size; also bounds the Impala encoder's 84x84 activations, which OOM
    # on 8GB GPUs at 512.
    config.args_buffer.sample_batch_size = batch_size
    config.args_buffer.sample_sequence_length = 4  # 3-step returns, as the paper's M-IQN
    config.args_buffer.period = 1
    config.args_buffer.min_length = 1000
    config.args_buffer.max_size = 1000000  # in transitions
    config.args_buffer.priority_exponent = 0.2  # BTR
    config.args_buffer.observation_shape = observation_shape
    config.args_buffer.time_step_dtypes = time_step_dtypes = TimeStepDQNDtypesAtari

    config.state_recover = state_recover = restore_dqn_flax_state
    config.args_state_recover = args_state_recover = ml_collections.ConfigDict()
    config.args_state_recover.rng = None
    config.args_state_recover.network = network
    config.args_state_recover.args_network = args_network
    config.args_state_recover.optimizer = optimizer
    config.args_state_recover.args_optimizer = args_optimizer
    config.args_state_recover.observation_shape = observation_shape
    # tau is unused: target_update_period > 0 selects BTR's hard-copy path instead
    # of Polyak blending (see TrainState.apply_gradients in agents/miqn.py).
    config.args_state_recover.tau = 0.005
    config.args_state_recover.target_update_period = 500  # BTR: hard target copy
    config.args_state_recover.checkpointdir = checkpointdir

    config.trainer = TrainerSequential
    config.args_trainer = ml_collections.ConfigDict()
    config.args_trainer.trainer_seed = trainer_seed
    config.args_trainer.log_level = log_level
    config.args_trainer.workdir = workdir
    config.args_trainer.env_frames = env_frames
    config.args_trainer.replay_ratio = replay_ratio
    config.args_trainer.summary_writing_period = 100  # in gradient updates
    config.args_trainer.save_period = 1000  # orbax checkpointing, in gradient updates
    config.args_trainer.state_recover = state_recover
    config.args_trainer.args_state_recover = args_state_recover
    config.args_trainer.get_update_step = get_update_step
    config.args_trainer.args_get_update_step = ml_collections.ConfigDict()
    config.args_trainer.args_get_update_step.apply_fn = None
    config.args_trainer.args_get_update_step.gamma = 0.997  # BTR
    config.args_trainer.args_get_update_step.entropy_temperature = 0.03
    config.args_trainer.args_get_update_step.munchausen_coefficient = 0.9
    config.args_trainer.args_get_update_step.clip_value_min = -1.0
    config.args_trainer.args_get_update_step.quantile_huber_kappa = 1.0
    # BTR's own N = N' = K = 8 (see docs/miqn.md for the previously-observed flat-Q
    # failure mode risk at this sample count with M-IQN).
    config.args_trainer.args_get_update_step.num_tau_samples = 8
    config.args_trainer.args_get_update_step.num_tau_prime_samples = 8
    config.args_trainer.args_get_update_step.num_quantile_samples = 8
    config.args_trainer.args_get_update_step.max_abs_reward = 1.0
    # BTR's own per_beta_anneal defaults to off, so its IS exponent is a fixed 0.45,
    # not annealed.
    config.args_trainer.args_get_update_step.importance_weight_scheduler_fn = (
        optax.constant_schedule(0.45)
    )
    config.args_trainer.args_get_update_step.obs_preprocess_fn = lambda x: (
        x.astype(jnp.bfloat16) / jnp.bfloat16(255.0)
    )
    config.args_trainer.get_update_epoch = get_update_epoch
    config.args_trainer.args_get_update_epoch = ml_collections.ConfigDict()
    config.args_trainer.args_get_update_epoch.update_step_fn = None
    config.args_trainer.args_get_update_epoch.buffer_lock = None  # injected by trainer
    config.args_trainer.args_get_update_epoch.buffer = None  # injected by trainer
    config.args_trainer.buffer = buffer
    config.args_trainer.args_buffer = args_buffer

    config.args_trainer.collector = CollectorDQNUniform
    config.args_trainer.args_collector = ml_collections.ConfigDict()
    config.args_trainer.args_collector.controller = None  # injected by trainer
    config.args_trainer.args_collector.trainer = None  # unused in the sequential path
    config.args_trainer.args_collector.collectors_seed = collectors_seed
    config.args_trainer.args_collector.log_level = log_level
    config.args_trainer.args_collector.report_period = 1000000  # _collecting never runs here
    config.args_trainer.args_collector.state_recover = state_recover
    config.args_trainer.args_collector.args_state_recover = args_state_recover
    config.args_trainer.args_collector.env = env
    config.args_trainer.args_collector.args_env = args_env
    config.args_trainer.args_collector.time_step_dtypes = time_step_dtypes
    # 1 = exact BTR interleaving (one 32-env step batch per learn() call, matching
    # replay_ratio above); raise it to trade interleaving fidelity for wall-clock speed.
    config.args_trainer.args_collector.steps_per_rollout = 1
    config.args_trainer.args_collector.get_select_action_fn = get_select_action_batch_fn
    config.args_trainer.args_collector.args_get_select_action_fn = ml_collections.ConfigDict()
    config.args_trainer.args_collector.args_get_select_action_fn.apply_fn = None
    config.args_trainer.args_collector.args_get_select_action_fn.num_quantile_samples = 8  # BTR
    # BTR combines NoisyNets with annealed eps-greedy for the first half of training,
    # then relies on noisy weights alone: eps 1.0 -> 0.01 over 2,000,000 frames (BTR's
    # own eps_steps), held at 0.01 until env_frames // 2, then disabled (eps = 0.0).
    config.args_trainer.args_collector.args_get_select_action_fn.epsilon_scheduler_fn = (
        optax.join_schedules(
            schedules=[
                optax.linear_schedule(init_value=1.0, end_value=0.01, transition_steps=2_000_000),
                optax.constant_schedule(0.01),
                optax.constant_schedule(0.0),
            ],
            boundaries=[2_000_000, env_frames // 2],
        )
    )
    config.args_trainer.args_collector.args_get_select_action_fn.obs_preprocess_fn = lambda x: (
        x.astype(jnp.float32) / 255.0
    )

    return config.lock()
