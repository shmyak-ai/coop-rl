import random

import tensorflow as tf
import numpy as np
import reverb
import ray
from ray.util.queue import Empty

from coop_rl.members import Worker


class DQNCollector(Worker):

    def __init__(self, run_config, exchange_actor, weights, collector_id):
        super().__init__(run_config, exchange_actor, weights)

        # to add data to the replay buffer
        self._buffer_client = reverb.Client(
            f'{run_config.buffer_server_ip}:{run_config.buffer_server_port}'
        )
        self._table_names = run_config.table_names

        self._collector_id = collector_id

    def _policy(self, obsns, epsilon, info):
        if np.random.rand() < epsilon:
            # the first step after reset is arbitrary
            if info is None:
                available_actions = [0, 1, 2, 3]
                actions = [random.choice(available_actions) for _ in range(self._n_players)]
            # other random actions are within actions != opposite to the previous ones
            else:
                actions = [random.choice(info[i]['allowed_actions']) for i in range(self._n_players)]
            return actions
        else:
            # it receives observations for all geese and predicts best actions one by one
            best_actions = []
            obsns = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obsns)
            for i in range(self._n_players):
                obs = obsns[i]
                Q_values = self._predict(obs)
                best_actions.append(np.argmax(Q_values[0]))
            return best_actions

    def do_collect(self):
        num_collects = 0
        num_updates = 0

        while True:
            # trainer will switch to done on the last iteration
            is_done = ray.get(self._workers_info.get_done.remote())
            if is_done:
                # print("Collecting is done.")
                return num_collects, num_updates
            # get the current turn, so collectors (workers) update weights one by one
            curr_worker = ray.get(self._workers_info.get_global_v.remote())
            # check the current turn
            if curr_worker == self._worker_id:
                if not self._ray_queue.empty():  # see below
                    try:
                        # block = False will cause an exception if there is no data in the queue,
                        # which is not handled by a ray queue (incompatibility with python 3.8 ?)
                        weights = self._ray_queue.get(block=False)
                        if curr_worker == self._num_collectors:
                            # print(f"Worker {curr_worker} updates weights")
                            ray.get(self._workers_info.set_global_v.remote(1))
                            num_updates += 1
                        elif curr_worker < self._num_collectors:
                            ray.get(self._workers_info.set_global_v.remote(curr_worker + 1))
                            # print(f"Worker {curr_worker} update weights")
                            num_updates += 1
                        else:
                            print("Wrong worker")
                            raise NotImplementedError
                    except Empty:
                        weights = None
                else:
                    weights = None
            else:
                weights = None

            if weights is not None:
                self._model.set_weights(weights)
                # print("Weights are updated")

            epsilon = None
            # t1 = time.time()
            if self._data is not None:
                if num_collects % 25 == 0:
                    self._collect(epsilon, is_random=True)
                    # print("Episode with a random trajectory was collected; "
                    #       f"Num of collects: {num_collects}")
                else:
                    self._collect(epsilon)
                    # print(f"Num of collects: {num_collects}")
            else:
                if num_collects < 10000 or num_collects % 25 == 0:
                    if num_collects == 9999:
                        print("Collector: The last initial random collect.")
                    self._collect(epsilon, is_random=True)
                else:
                    self._collect(epsilon)
            num_collects += 1
            # print(f"Num of collects: {num_collects}")
            # t2 = time.time()
            # print(f"Collecting. Time: {t2 - t1}")
