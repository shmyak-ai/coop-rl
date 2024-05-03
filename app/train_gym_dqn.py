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

"""DQN with a Gym Cartpole environment Hyperparameter configuration."""

import ray
import tensorflow as tf

from coop_rl.agents.dqn import JaxDQNAgent
from coop_rl.configs.dqn_cartpole import get_config
from coop_rl.workers.collectors import DQNCollector
from coop_rl.workers.evaluators import Evaluator
from coop_rl.workers.exchange_actors import (
    ControlActor,
    ReplayActor,
)
from coop_rl.utils import check_environment


def complex_call():

    conf = get_config()
    conf.observation_shape, conf.observation_dtype, conf.num_actions = check_environment(conf)

    # trainer, buffer actor, evaluator + collectors
    parallel_calls = 3 + conf.num_collectors
    is_gpu = bool(tf.config.list_physical_devices('GPU'))
    if conf.debug:
        ray.init(local_mode=True)
    elif is_gpu:
        ray.init(num_cpus=parallel_calls - 1, num_gpus=1)
    else:
        ray.init(num_cpus=parallel_calls)

    # make python classes ray actors
    agent_object = JaxDQNAgent
    if is_gpu:
        trainer_remote = ray.remote(num_gpus=1)(agent_object)
    else:
        trainer_remote = ray.remote(agent_object)
    collector_remotes = [ray.remote(DQNCollector) for _ in range(conf.num_collectors)]
    evaluator_remote = ray.remote(Evaluator)

    # initialization
    control_actor = ControlActor.remote(conf.num_collectors)
    replay_actor = ReplayActor.remote(conf.replay)
    trainer_agent = trainer_remote.remote(
        **conf.agent,
        control_actor=control_actor,
        replay_actor=replay_actor,
    )
    collector_agents = []
    for i, collector_remote in enumerate(collector_remotes):
        collector_agents.append(collector_remote.remote(
            **conf.collector,
            exchange_actor=exchange_actor,
            weights=data_net,
            collector_id=i + 1,
            ))
    evaluator_agents = []
    for evaluator_object in evaluator_objects:
        evaluator_agents.append(evaluator_object.remote(
            conf,
            exchange_actor=exchange_actor,
            weights=data_net,
            ))

    # remote calls
    trainer_futures = [agent.training.remote() for agent in trainer_agents]
    collect_info_futures = [agent.collecting.remote() for agent in collector_agents]
    eval_info_futures = [agent.evaluating.remote() for agent in evaluator_agents]

    # get results
    outputs = ray.get(trainer_futures)
    collect_info = ray.get(collect_info_futures)
    print(f"Collect info: {collect_info}")
    _ = ray.get(eval_info_futures)

    _, _, checkpoint = outputs[0]
    with open('data/checkpoint', 'w') as text_file:
        print(checkpoint, file=text_file)

    ray.shutdown()
    print("Done")


if __name__ == '__main__':
    complex_call()
