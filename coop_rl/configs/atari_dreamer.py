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

import hashlib
import time
from pathlib import Path

import elements
import ml_collections
import neptune
import ruamel.yaml as yaml
from ml_collections import config_dict

from coop_rl.agents.dreamer import get_select_action_fn, get_update_epoch, get_update_step, restore_dreamer_flax_state
from coop_rl.base_types import AtariTimeStepDtypes
from coop_rl.buffers import BufferTrajectoryDreamer
from coop_rl.environment import HandlerEnvDreamerAtari
from coop_rl.workers.auxiliary import Controller
from coop_rl.workers.collectors import DQNCollectorUniform
from coop_rl.workers.trainers import Trainer

DREAMER_CONFIG_PATH = Path(__file__).resolve().parent / "dreamer.yaml"
DREAMER_ARGV = [
    "--configs",
    "debug",
    "--logdir",
    "/home/sia/dreamer_results/debug",
    "--task",
    "atari_pong",
]


def get_dreamer_config():
    configs = elements.Path(DREAMER_CONFIG_PATH).read()
    configs = yaml.YAML(typ="safe").load(configs)
    parsed, other = elements.Flags(configs=["defaults"]).parse_known(DREAMER_ARGV)
    config = elements.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
    config = config.update(logdir=(config.logdir.format(timestamp=elements.timestamp())))
    return elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
        task=config.task,
        env=config.env,
    )


def get_config():
    config = ml_collections.ConfigDict()

    log_level = config_dict.FieldReference("INFO", field_type=str)
    observation_shape = config_dict.FieldReference(None, field_type=dict)
    observation_dtype = config_dict.FieldReference(None, field_type=dict)
    actions_shape = config_dict.FieldReference(None, field_type=dict)
    workdir = config_dict.FieldReference(None, field_type=str)
    checkpointdir = config_dict.FieldReference(None, field_type=str)
    dreamer_config = config_dict.FieldReference(None, field_type=elements.config.Config)

    seed = 73
    buffer_seed, trainer_seed, collectors_seed = seed + 1, seed + 2, seed + 3
    steps = 3000000
    training_iterations_per_step = 1

    config.neptune_run = neptune_run = neptune.init_run
    config.args_neptune_run = args_neptune_run = ml_collections.ConfigDict()
    config.args_neptune_run.custom_run_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    config.args_neptune_run.project = "sha/coop-rl"
    config.args_neptune_run.name = "dreamer"
    config.args_neptune_run.monitoring_namespace = "monitoring"

    config.log_level = log_level
    config.num_collectors = num_collectors = 5
    config.num_samplers = 1
    config.observation_shape = observation_shape
    config.observation_dtype = observation_dtype
    config.actions_shape = actions_shape
    config.workdir = workdir
    config.dreamer_config = dreamer_config
    config.dreamer_config = get_dreamer_config()  # only to prevent type change of elements.config.Config

    config.env = env = HandlerEnvDreamerAtari
    config.args_env = args_env = ml_collections.ConfigDict()
    config.args_env.dreamer_config = dreamer_config

    config.buffer = buffer = BufferTrajectoryDreamer
    config.args_buffer = args_buffer = ml_collections.ConfigDict()
    config.args_buffer.buffer_seed = buffer_seed
    config.args_buffer.add_batch_size = num_collectors
    config.args_buffer.sample_batch_size = 32
    config.args_buffer.sample_sequence_length = 3
    config.args_buffer.period = 1
    config.args_buffer.min_length = 100
    config.args_buffer.max_size = 300000  # in transitions
    config.args_buffer.observation_shape = observation_shape
    config.args_buffer.time_step_dtypes = time_step_dtypes = AtariTimeStepDtypes()

    config.state_recover = state_recover = restore_dreamer_flax_state
    config.args_state_recover = args_state_recover = ml_collections.ConfigDict()
    config.args_state_recover.dreamer_config = dreamer_config
    config.args_state_recover.observation_shape = observation_shape
    config.args_state_recover.actions_shape = actions_shape
    config.args_state_recover.checkpointdir = checkpointdir

    config.agent_params = agent_params = None

    config.controller = Controller

    config.trainer = Trainer
    config.args_trainer = ml_collections.ConfigDict()
    config.args_trainer.trainer_seed = trainer_seed
    config.args_trainer.log_level = log_level
    config.args_trainer.workdir = workdir
    config.args_trainer.steps = steps
    config.args_trainer.training_iterations_per_step = training_iterations_per_step
    config.args_trainer.summary_writing_period = 100  # logging and reporting
    config.args_trainer.save_period = 10000  # orbax checkpointing
    config.args_trainer.synchronization_period = 10  # send params to control actor
    config.args_trainer.state_recover = state_recover
    config.args_trainer.args_state_recover = args_state_recover
    config.args_trainer.get_update_step = get_update_step
    config.args_trainer.get_update_epoch = get_update_epoch
    config.args_trainer.agent_params = agent_params
    config.args_trainer.buffer = buffer
    config.args_trainer.args_buffer = args_buffer
    config.args_trainer.neptune_run = neptune_run
    config.args_trainer.args_neptune_run = args_neptune_run
    config.args_trainer.num_samples_on_gpu_cache = 100
    config.args_trainer.num_samples_to_gpu = 50
    config.args_trainer.num_semaphor = 1

    config.collector = DQNCollectorUniform
    config.args_collector = ml_collections.ConfigDict()
    config.args_collector.collectors_seed = collectors_seed
    config.args_collector.log_level = log_level
    config.args_collector.report_period = 100  # per rollouts sampled
    config.args_collector.state_recover = state_recover
    config.args_collector.args_state_recover = args_state_recover
    config.args_collector.env = env
    config.args_collector.args_env = args_env
    config.args_collector.neptune_run = neptune_run
    config.args_collector.args_neptune_run = args_neptune_run
    config.args_collector.get_select_action_fn = get_select_action_fn
    config.args_collector.time_step_dtypes = time_step_dtypes

    return config
