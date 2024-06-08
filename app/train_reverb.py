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
import os
import tempfile
import time
from pathlib import Path

import ray
from configs.reverb import atari

configs = {
    "atari": atari,
    "classic_control": None,
}


def main():
    parser = argparse.ArgumentParser(description="Cooperative reinforcement learning.")
    parser.add_argument("--mode", required=True, type=str, choices=("local", "distributed"))
    parser.add_argument("--config", required=True, type=str, choices=("classic_control", "atari"))
    parser.add_argument(
        "--workdir",
        type=str,
        default=os.path.join(Path.home(), "coop-rl_results/"),
        help="Path to the tensorboard logs, checkpoints, etc.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.workdir):
        os.mkdir(args.workdir)
    workdir = tempfile.mkdtemp(prefix=args.workdir)
    print(f"Workdir is {workdir}.")

    conf = configs[args.config].get_config()
    conf.observation_shape, conf.observation_dtype, conf.num_actions = conf.args_collector.handler_env.check_env(
        conf.env_name, conf.stack_size
    )
    conf.workdir = workdir

    if args.mode == "local":
        reverb_server = conf.reverb_server(**conf.args_reverb_server)  # noqa: F841
        conf.table_name = reverb_server.table_name

        collector = conf.collector(
            **conf.args_collector,
        )
        trainer = conf.agent(
            **conf.args_agent,
        )
        collector.collecting_reverb(3)
        trainer.training_reverb()

        print("Done.")
    elif args.mode == "distributed":
        # collectors, agent, replay actor use cpus
        ray.init(num_cpus=conf.num_collectors + 2, num_gpus=1)

        conf.control_actor = ray.remote(num_cpus=0)(conf.control_actor)
        conf.replay_actor = ray.remote(num_cpus=1)(conf.replay_actor)
        conf.collector = ray.remote(num_cpus=1)(conf.collector)
        conf.agent = ray.remote(num_cpus=1, num_gpus=1)(conf.agent)

        # initialization
        control_actor = conf.control_actor.remote()
        replay_actor = conf.replay_actor.remote(**conf.args_replay)
        collector_agents = [
            conf.collector.remote(
                collector_id=100 * i,
                **conf.args_collector,
                control_actor=control_actor,
                replay_actor=replay_actor,
            )
            for i in range(1, conf.num_collectors + 1)
        ]
        trainer_agent = conf.agent.remote(
            **conf.args_agent,
            control_actor=control_actor,
            replay_actor=replay_actor,
        )

        # remote calls
        collect_info_futures = [agent.collecting_remote.remote() for agent in collector_agents]
        trainer_futures = trainer_agent.training_remote.remote()
        # eval_info_futures = [agent.evaluating.remote() for agent in evaluator_agents]

        # get results
        ray.get(collect_info_futures)
        ray.get(trainer_futures)
        # ray.get(eval_futures)
        # ray.get(control_actor.set_done.remote())
        # print(f"3. Add count in the buffer: {ray.get(replay_actor.add_count.remote())}")
        time.sleep(1)

        ray.shutdown()
        print("Done; ray shutdown.")


if __name__ == "__main__":
    main()
