import random

import tensorflow as tf
import numpy as np
import reverb
import ray

from coop_rl.members import Worker


class DQNCollector(Worker):

    def __init__(self, run_config, exchange_actor, weights, *args, **kwargs):
        super().__init__(run_config, exchange_actor, weights)

        # to add data to the replay buffer
        self._buffer_client = reverb.Client(
            f'{run_config.buffer_server_ip}:{run_config.buffer_server_port}'
        )
        self._table_names = run_config.table_names
        self._epsilon = run_config.epsilon

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

    def _collect_trajectories_from_episode(self, epsilon):
        """
        Collects trajectories (items) to a buffer. For example, we have 4 points to collect: 1, 2, 3, 4
        the function will store trajectories:
        1, 2; 2, 3; 3, 4;
        1, 2, 3; 2, 3, 4;
        1, 2, 3, 4;
        The first table will be the largest, the second one 1 point smaller, etc.

        A buffer contains items, each item consists of several n_points;
        for a regular TD update an item should have 2 n_points (a minimum number to form one time step).
        One n_point contains (action, obs, reward, done);
        action, reward, done are for the current observation (or obs);
        e.g. action led to the obs, reward prior to the obs, if is it done at the current obs.

        this implementation creates writers for each player (goose) and stores
        n_points trajectories for all of them

        if epsilon is None assume an off policy gradient method where policy_logits required
        """

        # initialize writers for all players
        writers = [self._replay_memory_client.writer(max_sequence_length=self._n_points)
                   for _ in range(self._n_players)]
        obs_records = []
        info = None

        obsns = self._train_env.reset()
        action, reward, done = tf.constant(-1), tf.constant(0.), tf.constant(0.)
        obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)
        for i, writer in enumerate(writers):
            obs = obsns[i][0], obsns[i][1]
            obs_records.append(obs)
            if epsilon is None:
                policy_logits = tf.constant([0., 0., 0., 0.])
                writer.append((action, policy_logits, obs, reward, done))
            else:
                writer.append((action, obs, reward, done))

        # for step in it.count(0):
        while True:
            if epsilon is None:
                actions, policy_logits = self._policy(obs_records)
                policy_logits = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32),
                                                      policy_logits)
            else:
                actions = self._policy(obs_records, epsilon, info)
            obs_records = []
            # environment step receives actions and outputs observations for the dead players also
            # but it takes no effect
            obsns, rewards, dones, info = self._train_env.step(actions)
            actions = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.int32), actions)
            rewards = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), rewards)
            dones = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), dones)
            obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)
            for i, writer in enumerate(writers):
                action, reward, done = actions[i], rewards[i], dones[i]
                obs = obsns[i][0], obsns[i][1]
                obs_records.append(obs)
                try:
                    if epsilon is None:
                        writer.append((action, policy_logits[i], obs, reward, done))
                    else:
                        writer.append((action, obs, reward, done))  # returns Runtime Error if a writer is closed
                    # if step >= start_itemizing:
                    for steps in range(2, self._n_points + 1):
                        try:
                            writer.create_item(table=self._table_names[steps - 2], num_timesteps=steps, priority=1.)
                        except ValueError:
                            # stop new items creation if there are not enough buffered timesteps
                            break
                    if done:
                        writer.close()
                except RuntimeError:
                    # continue writing with a next writer if a current one is closed
                    continue
            if all(dones):
                break

    def collecting(self):
        num_collects = 0
        epsilon = self._epsilon

        while True:
            # a trainer will switch to done at the last iteration
            is_done = ray.get(self._exchange_actor.is_done.remote())
            if is_done:
                return num_collects

            self._model.set_weights(ray.get(self._exchange_actor.get_weights()))
            self._collect(epsilon)
            num_collects += 1
