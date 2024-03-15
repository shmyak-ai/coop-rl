import pickle
from pathlib import Path
from dataclasses import dataclass

import ray
import tensorflow as tf
import reverb
# from ray.util.queue import Queue

from coop_rl.agents_dqn import DQNAgent
from coop_rl.collectors import DQNCollector
from coop_rl.evaluators import Evaluator
from coop_rl.buffer import DQNUniformBuffer
from coop_rl.miscellaneous import ExchangeActor
from coop_rl.miscellaneous import check_environment


@dataclass
class RunConfig:
    # general
    debug: bool = True
    seed: int | None = None
    print_interval: int = 100
    save_interval: int = 100
    weights_update_interval: int = 100
    target_model_update_interval: int = 100

    # cluster
    num_trainers: int = 1
    num_evaluators: int = 1
    num_collectors: int = 1

    # dm reverb buffer
    buffer_server_ip: str | int = 'localhost'
    buffer_server_port: int = 8000
    batch_size: int = 64
    buffer_size: int = 500000
    tables_number: int = 5
    table_names: tuple[str] = tuple(f"uniform_table_{i}" for i in range(tables_number))

    # coop_rl: env, model and dataset obs(input) shapes should be compatible
    env_name: str = 'CartPole-v1'
    dataset: str = '1d'
    learning_rate: float = 1e-5
    optimizer: str = 'adam'
    loss: str = 'huber'
    # model
    model: str = 'dense_critic'
    n_features: int = 1024
    n_layers: int = 3
    # coop_rl to be updated from env
    input_shape: tuple[int] | None = None
    n_outputs: int | None = None


def complex_call():
    try:
        with open(Path.home() / 'coop-rl-data' / 'data.pickle', 'rb') as file:
            data_net = pickle.load(file)
    except FileNotFoundError:
        data_net = None

    try:
        reverb_checkpoint = open(Path.home() / 'coop-rl-data' / 'checkpoint').read()
    except FileNotFoundError:
        reverb_checkpoint = None

    conf = RunConfig()
    conf.input_shape, conf.n_outputs = check_environment(conf)

    parallel_calls = conf.num_trainers + conf.num_collectors + conf.num_evaluators
    is_gpu = bool(tf.config.list_physical_devices('GPU'))
    if conf.debug:
        ray.init(local_mode=True)
    elif is_gpu:
        ray.init(num_cpus=parallel_calls - 1, num_gpus=1)
    else:
        ray.init(num_cpus=parallel_calls)
    # queue = Queue(maxsize=100)  # interprocess queue to store recent model weights

    if reverb_checkpoint is not None:
        path = str(Path(reverb_checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None

    # creates a reverb replay server at the current node
    # specify a node in the cluster explicitely?
    # a port can be specified
    buffer = DQNUniformBuffer(  # noqa: F841
        run_config=conf,
        checkpointer=checkpointer
        )

    # global variable to control getting items order from the interprocess queue, a done condition
    # store weight for an evaluator
    exchange_actor = ExchangeActor.remote()

    agent_object = DQNAgent
    if is_gpu:
        trainer_objects = [
            ray.remote(num_gpus=1 / conf.num_trainers)(agent_object) for _ in range(conf.num_trainers)
            ]
    else:
        trainer_objects = [ray.remote(agent_object) for _ in range(conf.num_trainers)]
    collector_objects = [ray.remote(DQNCollector) for _ in range(conf.num_collectors)]
    evaluator_objects = [ray.remote(Evaluator) for _ in range(conf.num_evaluators)]

    # initialization
    trainer_agents = []
    for i, trainer_object in enumerate(trainer_objects):
        make_checkpoint = True if i == 0 else False  # make a buffer checkpoint only in the first worker
        trainer_agents.append(trainer_object.remote(
            conf,
            exchange_actor=exchange_actor,
            weights=data_net,
            make_checkpoint=make_checkpoint,
            ))
    collector_agents = []
    for i, collector_object in enumerate(collector_objects):
        collector_agents.append(collector_object.remote(
            conf,
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
