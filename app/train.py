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

"""The entry point to launch ray distributed training."""

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

import ray

from coop_rl.configs import atari_dqn

configs = {
    "atari": atari_dqn,
    "classic_control": None,
}

runtime_env_cpu = {
    "env_vars": {
        "JAX_PLATFORMS": "cpu",
    }
}
runtime_env_debug = {
    "env_vars": {
        "RAY_DEBUG_POST_MORTEM": "1",
    }
}


def main():
    parser = argparse.ArgumentParser(description="Cooperative reinforcement learning.")
    parser.add_argument("--debug-ray", action="store_true", help="Ray debug environment activation.")
    parser.add_argument("--debug-log", action="store_true", help="Enable debug logs.")
    parser.add_argument("--config", required=True, type=str, choices=("classic_control", "atari"))
    parser.add_argument(
        "--workdir",
        type=str,
        default=os.path.join(Path.home(), "coop-rl_results/"),
        help="Path to the tensorboard logs, checkpoints, etc.",
    )
    parser.add_argument("--orbax-checkpoint-dir", type=str, help="The absolute path to the orbax checkpoint dir")

    args = parser.parse_args()

    if args.debug_ray:
        ray.init(runtime_env=runtime_env_debug)
    else:
        ray.init()

    # call jax stuff after ray init - after forking
    conf = configs[args.config].get_config()
    conf.observation_shape, conf.observation_dtype, conf.num_actions = conf.args_collector.env.check_env(
        conf.env_name, conf.stack_size
    )

    logger = logging.getLogger(__name__)
    if args.debug_log:
        log_level = "DEBUG"
        logger.setLevel(log_level)
        conf.log_level = log_level

    if not os.path.exists(args.workdir):
        os.mkdir(args.workdir)
    workdir = tempfile.mkdtemp(prefix=args.workdir)
    logger.info(f"Workdir is {workdir}.")
    conf.workdir = workdir

    if args.orbax_checkpoint_dir is not None:
        conf.args_state_recover.checkpointdir = args.orbax_checkpoint_dir
        conf.args_collector.flax_state = conf.state_recover(**conf.args_state_recover)
        # update collector arguments if there is a state available
        conf.args_collector.epsilon_decay_period = 1
        conf.args_collector.warmup_steps = 1

    # Transform to remote objects
    # with 0 gpus and a regular runtime jax will complain about gpu devices
    conf.controller = ray.remote(num_cpus=0, num_gpus=0, runtime_env=runtime_env_cpu)(conf.controller)
    conf.buffer = ray.remote(num_cpus=1, num_gpus=0, runtime_env=runtime_env_cpu)(conf.buffer)
    conf.collector = ray.remote(num_cpus=1, num_gpus=0, runtime_env=runtime_env_cpu)(conf.collector)
    # conf.trainer = ray.remote(num_cpus=1, num_gpus=1)(conf.trainer)

    # initialization
    # we cannot put remote refs back to conf and cannot pass jax objects
    controller = conf.controller.remote()
    buffer = conf.buffer.remote(**conf.args_buffer)
    collectors = []
    for _ in range(conf.num_collectors):
        conf.args_collector.collectors_seed += 1
        collector = conf.collector.remote(**conf.args_collector, buffer=buffer, controller=controller)
        collectors.append(collector)
    # trainer = conf.trainer.remote(**conf.args_trainer, buffer=buffer, controller=controller)

    # remote calls
    collect_info_futures = [agent.collecting.remote() for agent in collectors]
    # trainer_futures = trainer.training.remote()

    # get results
    ray.get(collect_info_futures)
    # ray.get(trainer_futures)
    time.sleep(3)

    ray.shutdown()
    logger.info("Done; ray shutdown.")


if __name__ == "__main__":
    main()
