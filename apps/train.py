import pickle
from pathlib import Path

import numpy as np
import reverb


def complex_call(env_name, agent_name, data, checkpoint, plot=False):
    import ray
    from ray.util.queue import Queue

    num_trainers = 1
    num_evaluators = 1
    num_collectors = config["collectors"]
    parallel_calls = num_trainers + num_collectors + num_evaluators

    is_gpu = bool(tf.config.list_physical_devices('GPU'))
    if is_gpu:
        ray.init(num_cpus=parallel_calls, num_gpus=1)
    else:
        ray.init(num_cpus=parallel_calls)
    queue = Queue(maxsize=100)  # interprocess queue to store recent model weights
    # ray.init(local_mode=True)  # for debugging
    # queue = None  # queue does not work in debug mode

    if checkpoint is not None:
        path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None

    if config["buffer"] == "full_episode":
        # 1 table for an episode
        buffer = storage.UniformBuffer(num_tables=1,
                                       min_size=config["batch_size"], max_size=config["buffer_size"],
                                       checkpointer=checkpointer)
    else:
        # we need several tables for each step size
        buffer = storage.UniformBuffer(num_tables=config["n_points"] - 1,
                                       min_size=config["batch_size"], max_size=config["buffer_size"],
                                       checkpointer=checkpointer)

    agent_object = AGENTS[agent_name]

    if is_gpu:
        trainer_objects = [ray.remote(num_gpus=1 / num_trainers)(agent_object) for _ in range(num_trainers)]
    else:
        trainer_objects = [ray.remote(agent_object) for _ in range(num_trainers)]
    collector_objects = [ray.remote(worker.Collector) for _ in range(num_collectors)]
    evaluator_objects = [ray.remote(worker.Evaluator) for _ in range(num_evaluators)]

    # global variable to control getting items order from the interprocess queue, a done condition
    # store weight for an evaluator
    workers_info = misc.GlobalVarActor.remote()

    # eval = worker.Evaluator(env_name, config, buffer.table_names, buffer.server_port,
    #                         workers_info=workers_info)
    # _, wins = eval.evaluate_episodes()

    # initialization
    trainer_agents = []
    for i, trainer_object in enumerate(trainer_objects):
        make_checkpoint = True if i == 0 else False  # make a buffer checkpoint only in the first worker
        trainer_agents.append(trainer_object.remote(env_name, config,
                                                    buffer.table_names, buffer.server_port,
                                                    data=data, make_checkpoint=make_checkpoint, ray_queue=queue,
                                                    workers_info=workers_info))

    collector_agents = []
    for i, collector_object in enumerate(collector_objects):
        collector_agents.append(collector_object.remote(env_name, config,
                                                        buffer.table_names, buffer.server_port,
                                                        data=data, make_checkpoint=False, ray_queue=queue,
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
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    try:
        init_checkpoint = open('data/checkpoint', 'r').read()
    except FileNotFoundError:
        init_checkpoint = None

    if config["setup"] == "multi":
        multi_call(config["environment"], config["agent"], init_data, init_checkpoint)
    elif config["setup"] == "complex":
        complex_call(config["environment"], config["agent"], init_data, init_checkpoint)
    elif config["setup"] == "single":
        one_call(config["environment"], config["agent"], init_data, init_checkpoint)
