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

# from coop_rl.agents.dqn import JaxDQNAgent
from coop_rl.configs.dqn_cartpole import get_config
from coop_rl.utils import check_environment
from coop_rl.workers.exchange_actors import (
    ControlActor,
    ReplayActor,
)


def complex_call():

    conf = get_config()
    conf.observation_shape, conf.observation_dtype, conf.num_actions = \
        check_environment(conf.environment_name)

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
    collector_remotes = [ray.remote(conf.collector) for _ in range(conf.num_collectors)]
    # agent_object = JaxDQNAgent
    # if is_gpu:
    #     trainer_remote = ray.remote(num_gpus=1)(agent_object)
    # else:
    #     trainer_remote = ray.remote(agent_object)
    # evaluator_remote = ray.remote(Evaluator)

    # initialization
    control_actor = ControlActor.remote(conf.observation_shape)
    replay_actor = ReplayActor.remote(conf)
    collector_agents = []
    for collector_remote in collector_remotes:
        collector_agents.append(collector_remote.remote(
            **conf.args_collector,
            control_actor=control_actor,
            replay_actor=replay_actor,
            ))
    # trainer_agent = trainer_remote.remote(
    #     **conf.agent,
    #     control_actor=control_actor,
    #     replay_actor=replay_actor,
    # )

    # remote calls
    collect_info_futures = [agent.run_one_episode.remote() for agent in collector_agents]
    # trainer_futures = trainer_agent.training.remote()
    # eval_info_futures = [agent.evaluating.remote() for agent in evaluator_agents]

    # get results
    ray.get(collect_info_futures)
    # outputs = ray.get(trainer_futures)
    # _ = ray.get(eval_info_futures)

    ray.shutdown()
    print("Done")


if __name__ == '__main__':
    complex_call()
