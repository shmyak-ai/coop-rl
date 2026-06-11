# Copyright 2025 The Coop RL Authors.
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
from ml_collections import config_dict

from coop_rl.agents.dreamer import (
    build_model,
    get_select_action_fn,
    get_update_epoch,
    get_update_step,
    restore_dreamer_flax_state,
)
from coop_rl.base.buffers import BufferTrajectoryDreamer
from coop_rl.base.environment import HandlerEnvDreamerAtari
from coop_rl.base.utils import make_dreamer_optimizer
from coop_rl.workers.auxiliary import Controller
from coop_rl.workers.collectors import CollectorDreamerUniform
from coop_rl.workers.trainers import Trainer

cd = ml_collections.ConfigDict


def _args_network():
    """DreamerV3 'size1m' model + actor-critic hyperparameters (Atari path).

    Consumed by ``build_model(args_network, obs_space, act_space)``. These are the
    upstream ``dreamer.yaml`` defaults with the ``size1m`` overrides applied
    (rssm deter=512/hidden=64/classes=4, depth=4, units=64).
    """
    c = ml_collections.ConfigDict()

    c.replay_context = 1

    # World model.
    c.dyn = ml_collections.ConfigDict()
    c.dyn.rssm = cd(
        dict(
            deter=512,
            hidden=64,
            stoch=32,
            classes=4,
            act="silu",
            norm="rms",
            unimix=0.01,
            outscale=1.0,
            imglayers=2,
            obslayers=1,
            dynlayers=1,
            blocks=8,
            free_nats=1.0,
        )
    )
    c.enc = ml_collections.ConfigDict()
    c.enc.simple = cd(
        dict(depth=4, mults=(2, 3, 4, 4), layers=3, units=64, act="silu", norm="rms", kernel=5)
    )
    c.dec = ml_collections.ConfigDict()
    c.dec.simple = cd(
        dict(
            depth=4,
            mults=(2, 3, 4, 4),
            layers=3,
            units=64,
            act="silu",
            norm="rms",
            outscale=1.0,
            kernel=5,
            bspace=8,
        )
    )

    # Heads.
    c.rewhead = cd(dict(layers=1, units=64, act="silu", norm="rms", bins=255, outscale=0.0))
    c.conhead = cd(dict(layers=1, units=64, act="silu", norm="rms", outscale=1.0))
    c.value = cd(dict(layers=3, units=64, act="silu", norm="rms", bins=255, outscale=0.0))
    c.policy = cd(
        dict(
            layers=3,
            units=64,
            act="silu",
            norm="rms",
            minstd=0.1,
            maxstd=1.0,
            outscale=0.01,
            unimix=0.01,
        )
    )
    c.policy_dist_disc = "categorical"
    c.policy_dist_cont = "bounded_normal"

    # Loss weights.
    c.loss_scales = cd(
        dict(rec=1.0, rew=1.0, con=1.0, dyn=1.0, rep=0.1, policy=1.0, value=1.0, repval=0.3)
    )

    # Actor-critic.
    c.ac_grads = False
    c.contdisc = True
    c.horizon = 333
    c.imag_length = 15
    c.imag_last = 0
    c.reward_grad = True
    c.repval_loss = True
    c.repval_grad = True
    c.imag_loss = cd(dict(slowtar=False, lam=0.95, actent=3e-4, slowreg=1.0))
    c.repl_loss = cd(dict(slowtar=False, lam=0.95, slowreg=1.0))
    c.retnorm = cd(dict(impl="perc", rate=0.01, limit=1.0, perclo=5.0, perchi=95.0, debias=False))
    c.valnorm = cd(dict(impl="none", rate=0.01, limit=1e-8))
    c.advnorm = cd(dict(impl="none", rate=0.01, limit=1e-8))

    return c


def get_config():
    config = ml_collections.ConfigDict()

    log_level = config_dict.FieldReference("INFO", field_type=str)
    observation_shape = config_dict.FieldReference(None, field_type=dict)
    observation_dtype = config_dict.FieldReference(None, field_type=dict)
    actions_shape = config_dict.FieldReference(None, field_type=dict)
    workdir = config_dict.FieldReference(None, field_type=str)
    checkpointdir = config_dict.FieldReference(None, field_type=str)

    seed = 73
    buffer_seed, trainer_seed, collectors_seed = seed + 1, seed + 2, seed + 3
    steps = 3000000
    training_iterations_per_step = 1

    # Run / batching.
    num_envs = 16
    batch_size = 16
    batch_length = 64
    slow_rate = 0.02  # slow-critic EMA rate (upstream slowvalue.rate)

    config.log_level = log_level
    config.num_collectors = 10
    config.num_samplers = 3
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.actions_shape = actions_shape
    config.workdir = workdir

    config.network = network = build_model
    config.args_network = args_network = _args_network()

    config.optimizer = optimizer = make_dreamer_optimizer
    config.args_optimizer = args_optimizer = ml_collections.ConfigDict()
    config.args_optimizer.lr = 4e-5
    config.args_optimizer.agc = 0.3
    config.args_optimizer.eps = 1e-20
    config.args_optimizer.beta1 = 0.9
    config.args_optimizer.beta2 = 0.999
    config.args_optimizer.momentum = True
    config.args_optimizer.wd = 0.0
    config.args_optimizer.schedule = "const"
    config.args_optimizer.warmup = 1000
    config.args_optimizer.anneal = 0

    config.env = env = HandlerEnvDreamerAtari
    config.args_env = args_env = ml_collections.ConfigDict()
    config.args_env.env_name = "ale_py:ALE/Pong-v5"
    config.args_env.num_envs = num_envs
    config.args_env.screen_size = 96
    config.args_env.sticky = True

    config.buffer = buffer = BufferTrajectoryDreamer
    config.args_buffer = args_buffer = ml_collections.ConfigDict()
    config.args_buffer.args_network = args_network
    config.args_buffer.buffer_seed = buffer_seed
    config.args_buffer.add_batch_size = num_envs
    config.args_buffer.sample_batch_size = batch_size
    config.args_buffer.sample_sequence_length = batch_length
    config.args_buffer.period = 1
    config.args_buffer.min_length = 1000
    config.args_buffer.max_size = 300000  # in transitions
    config.args_buffer.observation_shape = observation_shape
    config.args_buffer.actions_shape = actions_shape

    config.state_recover = state_recover = restore_dreamer_flax_state
    config.args_state_recover = args_state_recover = ml_collections.ConfigDict()
    config.args_state_recover.rng = None
    config.args_state_recover.network = network
    config.args_state_recover.args_network = args_network
    config.args_state_recover.optimizer = optimizer
    config.args_state_recover.args_optimizer = args_optimizer
    config.args_state_recover.observation_shape = observation_shape
    config.args_state_recover.actions_shape = actions_shape
    config.args_state_recover.batch_size = batch_size
    config.args_state_recover.batch_length = batch_length
    config.args_state_recover.slow_rate = slow_rate
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
    config.args_trainer.summary_writing_period = 100  # logging and reporting
    config.args_trainer.save_period = 1000  # orbax checkpointing
    config.args_trainer.synchronization_period = 10  # send params to control actor
    config.args_trainer.state_recover = state_recover
    config.args_trainer.args_state_recover = args_state_recover
    config.args_trainer.get_update_step = get_update_step
    config.args_trainer.args_get_update_step = ml_collections.ConfigDict()
    config.args_trainer.args_get_update_step.apply_fn = None
    config.args_trainer.get_update_epoch = get_update_epoch
    config.args_trainer.args_get_update_epoch = ml_collections.ConfigDict()
    config.args_trainer.args_get_update_epoch.update_step_fn = None
    config.args_trainer.args_get_update_epoch.buffer_lock = None
    config.args_trainer.args_get_update_epoch.buffer = None
    config.args_trainer.buffer = buffer
    config.args_trainer.args_buffer = args_buffer
    config.args_trainer.num_samples_on_gpu_cache = 3

    config.collector = CollectorDreamerUniform
    config.args_collector = ml_collections.ConfigDict()
    config.args_collector.controller = None
    config.args_collector.trainer = None
    config.args_collector.workdir = workdir
    config.args_collector.collectors_seed = collectors_seed
    config.args_collector.log_level = log_level
    config.args_collector.report_period = 1  # per rollouts sampled
    config.args_collector.state_recover = state_recover
    config.args_collector.args_state_recover = args_state_recover
    config.args_collector.env = env
    config.args_collector.args_env = args_env
    config.args_collector.get_select_action_fn = get_select_action_fn

    return config.lock()
