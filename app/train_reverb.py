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

"""The entry point to launch local / ray distributed training."""

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

import ray
import reverb
from configs.reverb import atari

configs = {
    "atari": atari,
    "classic_control": None,
}

runtime_env_cpu = {
    "env_vars": {
        "JAX_PLATFORMS": "cpu",
    }
}
runtime_env_debug = {
    "env_vars": {
        "RAY_DEBUG": "1",
    }
}


def main():
    parser = argparse.ArgumentParser(description="Cooperative reinforcement learning.")
    parser.add_argument("--debug-ray", action="store_true", help="Ray debug environment activation.")
    parser.add_argument("--debug-log", action="store_true", help="Enable debug logs.")
    parser.add_argument("--mode", required=True, type=str, choices=("local", "distributed"))
    parser.add_argument("--config", required=True, type=str, choices=("classic_control", "atari"))
    parser.add_argument(
        "--workdir",
        type=str,
        default=os.path.join(Path.home(), "coop-rl_results/"),
        help="Path to the tensorboard logs, checkpoints, etc.",
    )
    parser.add_argument("--orbax-checkpoint-dir", type=str, help="The absolute path to the orbax checkpoint dir")
    parser.add_argument("--reverb-checkpoint-dir", type=str, help="The absolute path to the reverb checkpoint dir")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    log_level = "INFO"
    if args.debug_log:
        log_level = "DEBUG"
        logger.setLevel(log_level)

    if not os.path.exists(args.workdir):
        os.mkdir(args.workdir)
    workdir = tempfile.mkdtemp(prefix=args.workdir)
    logger.info(f"Workdir is {workdir}.")

    conf = configs[args.config].get_config()
    conf.observation_shape, conf.observation_dtype, conf.num_actions = conf.args_collector.handler_env.check_env(
        conf.env_name, conf.stack_size
    )
    conf.workdir = workdir
    if args.reverb_checkpoint_dir is not None:
        checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
            path=args.reverb_checkpoint_dir.rsplit("/", 1)[0]
        )
    else:
        checkpointer = None

    if args.mode == "local":
        if args.orbax_checkpoint_dir is not None:
            conf.args_state_recover.checkpointdir = args.orbax_checkpoint_dir
            flax_state = conf.state_recover(**conf.args_state_recover)
            # update collector arguments if there is a state available
            conf.args_collector.epsilon_decay_period = 1
            conf.args_collector.warmup_steps = 1
        else:
            flax_state = None
        reverb_server = conf.reverb_server(**conf.args_reverb_server, checkpointer=checkpointer)  # noqa: F841
        conf.table_name = reverb_server.table_name()
        collector = conf.collector(
            **conf.args_collector,
            control_actor=conf.control_actor(),
            trainer=conf.agent,
            args_trainer=conf.args_agent,
            state=flax_state,
        )
        collector.collecting_training()
        logger.info("Done.")
    elif args.mode == "distributed":
        # collectors, agent, replay actor use cpus
        if args.debug_ray:
            ray.init(runtime_env=runtime_env_debug)
        else:
            ray.init()

        # call jax stuff after ray init - after forking
        if args.orbax_checkpoint_dir is not None:
            conf.args_state_recover.checkpointdir = args.orbax_checkpoint_dir
            flax_state = conf.state_recover(**conf.args_state_recover)
            # update collector arguments if there is a state available
            conf.args_collector.epsilon_decay_period = 1
            conf.args_collector.warmup_steps = 1
        else:
            flax_state = None
        conf.control_actor = ray.remote(num_cpus=0, num_gpus=0, runtime_env=runtime_env_cpu)(conf.control_actor)
        conf.reverb_server = ray.remote(num_cpus=1, num_gpus=0, runtime_env=runtime_env_cpu)(conf.reverb_server)
        conf.collector = ray.remote(num_cpus=1, num_gpus=0, runtime_env=runtime_env_cpu)(conf.collector)
        conf.agent = ray.remote(num_cpus=1, num_gpus=1)(conf.agent)

        # initialization
        reverb_server = conf.reverb_server.remote(**conf.args_reverb_server, checkpointer=checkpointer)  # noqa: F841
        conf.table_name = ray.get(reverb_server.table_name.remote())
        control_actor = conf.control_actor.remote()
        trainer_agent = conf.agent.remote(
            **conf.args_agent,
            control_actor=control_actor,
            state=flax_state,
            log_level=log_level,
        )
        collector_agents = [
            conf.collector.remote(
                collector_id=100 * i,
                **conf.args_collector,
                control_actor=control_actor,
                state=flax_state,
                log_level=log_level,
            )
            for i in range(1, conf.num_collectors + 1)
        ]

        # remote calls
        collect_info_futures = [agent.collecting.remote() for agent in collector_agents]
        trainer_futures = trainer_agent.training.remote()
        # eval_info_futures = [agent.evaluating.remote() for agent in evaluator_agents]

        # get results
        ray.get(collect_info_futures)
        ray.get(trainer_futures)
        # ray.get(eval_futures)
        # ray.get(control_actor.set_done.remote())
        time.sleep(1)

        ray.shutdown()
        logger.info("Done; ray shutdown.")


if __name__ == "__main__":
    main()
