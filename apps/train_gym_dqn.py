import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List

import ray
import tensorflow as tf
import reverb
from ray.util.queue import Queue

from coop_rl.buffer import UniformBuffer
from coop_rl.agents_dqn import DQNAgent
from coop_rl.workers import (
    Collector,
    Evaluator
)
from coop_rl.misc import GlobalVarActor


@dataclass
class RunConfig:
    # cluster
    num_trainers: int = 1
    num_evaluators: int = 1
    num_collectors: int = 1

    # dm reverb buffer
    buffer_server_ip: str = 'localhost'
    buffer_server_port: str = '8000'
    batch_size: int = 64
    buffer_size: int = 500000
    tables_number: int = 5
    table_names: List(str) = [f"uniform_table_{i}" for i in range(tables_number)]

    # coop_rl
    model: str = 'dense_value'
    optimizer: str = 'adam'
    loss: str = 'huber'


def complex_call(data_net, reverb_checkpoint):

    try:
        with open(Path.home() / 'coop-rl-data' / 'data.pickle', 'rb') as file:
            data_net = pickle.load(file)
    except FileNotFoundError:
        data_net = None

    try:
        reverb_checkpoint = open(Path.home() / 'coop-rl-data' / 'checkpoint', 'r').read()
    except FileNotFoundError:
        reverb_checkpoint = None

    conf = RunConfig()
    queue = Queue(maxsize=100)  # interprocess queue to store recent model weights

    parallel_calls = conf.num_trainers + conf.num_collectors + conf.num_evaluators
    is_gpu = bool(tf.config.list_physical_devices('GPU'))
    if is_gpu:
        ray.init(num_cpus=parallel_calls, num_gpus=1)
    else:
        ray.init(num_cpus=parallel_calls)

    if reverb_checkpoint is not None:
        path = str(Path(reverb_checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None

    # we need several tables for each step size
    buffer = UniformBuffer(
        num_tables=conf.tables_number,
        table_names=conf.table_names,
        min_size=conf.batch_size,
        max_size=conf.buffer_size,
        checkpointer=checkpointer
        )

    agent_object = DQNAgent
    if is_gpu:
        trainer_objects = [
            ray.remote(num_gpus=1 / conf.num_trainers)(agent_object) for _ in range(conf.num_trainers)
            ]
    else:
        trainer_objects = [ray.remote(agent_object) for _ in range(conf.num_trainers)]
    collector_objects = [ray.remote(Collector) for _ in range(conf.num_collectors)]
    evaluator_objects = [ray.remote(Evaluator) for _ in range(conf.num_evaluators)]

    # global variable to control getting items order from the interprocess queue, a done condition
    # store weight for an evaluator
    workers_info = GlobalVarActor.remote()

    # initialization
    trainer_agents = []
    for i, trainer_object in enumerate(trainer_objects):
        make_checkpoint = True if i == 0 else False  # make a buffer checkpoint only in the first worker
        trainer_agents.append(trainer_object.remote(
            conf,
            data=data_net,
            make_checkpoint=make_checkpoint,
            ray_queue=queue,
            workers_info=workers_info
            ))

    collector_agents = []
    for i, collector_object in enumerate(collector_objects):
        collector_agents.append(collector_object.remote(env_name, config,
                                                        buffer.table_names, buffer.server_port,
                                                        data=data_net, make_checkpoint=False, ray_queue=queue,
                                                        worker_id=i + 1, workers_info=workers_info,
                                                        num_collectors=num_collectors))

    evaluator_agents = []
    for evaluator_object in evaluator_objects:
        evaluator_agents.append(evaluator_object.remote(env_name, config,
                                                        buffer.table_names, buffer.server_port,
                                                        workers_info=workers_info))

    # remote call
    trainer_futures = [agent.do_train.remote(iterations_number=config["iterations_number"],
                                             save_interval=config["save_interval"])
                       for agent in trainer_agents]

    collect_info_futures = [agent.do_collect.remote() for agent in collector_agents]
    eval_info_futures = [agent.do_evaluate.remote() for agent in evaluator_agents]

    # get results
    outputs = ray.get(trainer_futures)
    collect_info = ray.get(collect_info_futures)
    print(f"Collect info: {collect_info}")
    _ = ray.get(eval_info_futures)

    # rewards_array = np.empty(num_trainers)
    # steps_array = np.empty(num_trainers)
    # weights_list, mask_list = [], []
    # for count, (weights, mask, reward, steps, _) in enumerate(outputs):
    #     weights_list.append(weights)
    #     mask_list.append(mask)
    #     rewards_array[count] = reward
    #     steps_array[count] = steps
    #     print(f"Proc #{count}: Average reward = {reward:.2f}, Steps = {steps:.2f}")
    #     if plot:
    #         misc.plot_2d_array(weights[0], "Zero_lvl_with_reward_" + str(reward) + "_proc_" + str(count))
    #         misc.plot_2d_array(weights[2], "First_lvl_with_reward_" + str(reward) + "_proc_" + str(count))
    # argmax = rewards_array.argmax()
    # argmax = steps_array.argmax()
    # print(f"to save: Reward = {rewards_array[argmax]:.2f}, Steps = {steps_array[argmax]:.2f}")
    # data = {
    #     'weights': weights_list[argmax],
    #     'mask': mask_list[argmax],
    #     'reward': rewards_array[argmax]
    # }
    # with open('data/data.pickle', 'wb') as f:
    #     pickle.dump(data, f, protocol=4)

    _, _, checkpoint = outputs[0]
    with open('data/checkpoint', 'w') as text_file:
        print(checkpoint, file=text_file)

    ray.shutdown()
    print("Done")


if __name__ == '__main__':
    complex_call()
