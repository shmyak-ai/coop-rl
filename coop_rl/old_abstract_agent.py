import abc
import pickle
import itertools as it
import time

import numpy as np
import tensorflow as tf
import gym
import reverb
import ray


class Agent(abc.ABC):

    def __init__(self, env_name, config,
                 buffer_table_names, buffer_server_port,
                 data=None, make_checkpoint=False,
                 ray_queue=None, worker_id=None, workers_info=None, num_collectors=None
                 ):
        # environments; their hyperparameters
        self._train_env = gym.make(env_name)
        self._eval_env = gym.make(env_name)
        self._n_outputs = self._train_env.action_space.n  # number of actions
        space = self._train_env.observation_space
        self._n_players = len(space)  # number of players (geese)
        # env returns observations for all players, we need shape of any
        self._feature_maps_shape = space[0][0].shape  # height, width, channels; or layers, vectors
        self._scalar_features_shape = space[0][1].shape
        self._input_shape = (self._feature_maps_shape, self._scalar_features_shape)

        # data contains weighs, masks, and a corresponding reward
        self._data = data

        self._make_checkpoint = make_checkpoint

        # networks
        self._model = None
        self._target_model = None

        # hyperparameters for optimization
        self._default_lr = config["default_lr"]
        self._n_points = config["n_points"]
        self._sample_batch_size = config["batch_size"]
        self._data_cnt_ema = self._sample_batch_size * (self._n_points - 1)
        self._optimizer = config["optimizer"]
        self._loss_fn = config["loss"]

        # buffer
        self._table_names = buffer_table_names
        # an object with a client, which is used to store data on a server
        self._replay_memory_client = reverb.Client(f'localhost:{buffer_server_port}')

        self._is_full_episode = True if config["buffer"] == "full_episode" else False
        if self._is_full_episode:
            # a maximum number of points in geese environment
            self._collect = self._collect_episode
            self._items_sampled = [0, ]
        else:
            self._is_all_trajectories = config["all_trajectories"]
            if self._is_all_trajectories:
                self._collect = self._collect_trajectories_from_episode
            else:
                self._collect = self._collect_some_trajectories_from_episode
            self._items_sampled = [0 for _ in range(self._n_points - 1)]

        self._discount_rate = config["discount_rate"]

        self._start_epsilon = None
        self._final_epsilon = None
        self._is_policy_gradient = True if config["agent"] == "actor-critic" else False
        self._iterators = None
        # self._sampling_meter = 0

        self._ray_queue = ray_queue

        self._worker_id = worker_id
        self._workers_info = workers_info
        self._num_collectors = num_collectors

        if not config["debug"]:
            self._predict = tf.function(self._predict)

    def _predict(self, observation):
        return self._model(observation)

    def _policy(self, *args, **kwargs):
        raise NotImplementedError

    def _evaluate_episode(self, epsilon):
        """
        Epsilon 0 corresponds to greedy DQN _policy,
        if epsilon is None assume policy gradient _policy
        """
        obs_records = self._eval_env.reset()
        rewards_storage = np.zeros(self._n_players)
        for step in it.count(0):
            if epsilon is None:
                actions, _ = self._policy(obs_records)
            else:
                actions = self._policy(obs_records, epsilon, info=None)
            obs_records, rewards, dones, info = self._eval_env.step(actions)
            rewards_storage += np.asarray(rewards)
            if all(dones):
                break
        return rewards_storage.mean(), step

    def _evaluate_episodes(self, num_episodes=3, epsilon=None):
        episode_rewards = 0
        steps = 0
        for _ in range(num_episodes):
            reward, step = self._evaluate_episode(epsilon)
            episode_rewards += reward
            steps += step
        return episode_rewards / num_episodes, steps / num_episodes

    def _collect_episode(self, epsilon, is_random=False):
        """
        Collects an episode trajectory (1 item) to a buffer.

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
        # writers = [self._replay_memory_client.writer(max_sequence_length=self._n_points)
        #            for _ in range(self._n_players)]
        writers = [[] for _ in range(self._n_players)]
        ray_writers = [self._replay_memory_client.writer(max_sequence_length=self._n_points)
                       for _ in range(self._n_players)]

        dones = [False for _ in range(self._n_players)]  # for a first check
        ready = [False for _ in range(self._n_players)]
        ready_counter = [0 for _ in range(self._n_players)]

        obs_records = []
        info = None

        # some constants we are going to use repeatedly
        action_negative, reward_zero = tf.constant(-1), tf.constant(0.)
        done_true, done_false = tf.constant(1.), tf.constant(0.)
        policy_logits_zeros = tf.constant([0., 0., 0., 0.])
        rewards_saver = [None, None, None, None]
        obs_zeros = (tf.zeros(self._feature_maps_shape, dtype=tf.uint8),
                     tf.zeros(self._scalar_features_shape, dtype=tf.uint8))

        obsns = self._train_env.reset()
        obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)
        for i, writer in enumerate(writers):
            obs = obsns[i][0], obsns[i][1]
            obs_records.append(obs)
            if epsilon is None:
                writer.append((action_negative, policy_logits_zeros, obs, reward_zero, done_false))
            else:
                writer.append((action_negative, obs, reward_zero, done_false))
        step_counter = 1  # start with 1, since we have at least initialization
        # steps_per_worker_counter = [1 for _ in range(self._n_players)]

        while not all(ready):
            if not all(dones):
                if epsilon is None:
                    actions, policy_logits = self._policy(obs_records, is_random)
                    policy_logits = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32),
                                                          policy_logits)
                else:
                    actions = self._policy(obs_records, epsilon, info)

                step_counter += 1
                obsns, rewards, dones, info = self._train_env.step(actions)
                # environment step receives actions and outputs observations for the dead players also
                # but it takes no effect
                actions = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.int32), actions)
                rewards = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), rewards)
                dones = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), dones)
                obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)

            obs_records = []
            for i, writer in enumerate(writers):
                if ready_counter[i] == self._n_points - 1:
                    ready[i] = True
                    obs_records.append(obs_zeros)
                    continue
                action, reward, done = actions[i], rewards[i], dones[i]
                if done:
                    ready_counter[i] += 1
                    # the first 'done' encounter, save a final reward
                    if rewards_saver[i] is None:
                        rewards_saver[i] = reward
                        # steps_per_worker_counter[i] += 1
                    # consequent 'done' encounters, put zero actions and logits
                    else:
                        action = action_negative
                        if epsilon is None:
                            policy_logits[i] = policy_logits_zeros
                    obs = obs_zeros
                    # if 'done', store final rewards
                    reward = rewards_saver[i]
                else:
                    obs = obsns[i][0], obsns[i][1]
                    # steps_per_worker_counter[i] += 1
                obs_records.append(obs)
                if epsilon is None:
                    writer.append((action, policy_logits[i], obs, reward, done))
                else:
                    writer.append((action, obs, reward, done))  # returns Runtime Error if a writer is closed

        progress = tf.concat([tf.constant([0.]),
                              tf.linspace(0., 1., step_counter)[:-1],
                              tf.ones(self._n_points - 2)], axis=0)
        for i, ray_writer in enumerate(ray_writers):
            steps = len(writers[i])
            # progress = tf.concat([tf.constant([0.]),
            #                       tf.linspace(0., 1., steps_per_worker_counter[i])[:-1],
            #                       tf.ones(steps - steps_per_worker_counter[i])], axis=0)

            for step in range(steps):
                action, logits, obs, reward, done = (writers[i][step][0], writers[i][step][1],
                                                     writers[i][step][2], writers[i][step][3],
                                                     writers[i][step][4])
                ray_writer.append((action, logits, obs, reward, done, rewards_saver[i], progress[step]))
                if step >= self._n_points - 1:
                    ray_writer.create_item(table=self._table_names[0], num_timesteps=self._n_points, priority=1.)

            ray_writer.close()

    def _collect_episode_legacy(self, epsilon):
        """
        Collects an episode trajectory (1 item) to a buffer.

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
        # writers = [self._replay_memory_client.writer(max_sequence_length=self._n_points)
        #            for _ in range(self._n_players)]
        writers = [[] for _ in range(self._n_players)]
        ray_writers = [self._replay_memory_client.writer(max_sequence_length=self._n_points)
                       for _ in range(self._n_players)]
        dones = [False for _ in range(self._n_players)]  # for a first check
        obs_records = []
        info = None

        # some constants we are going to use repeatedly
        action_negative, reward_zero = tf.constant(-1), tf.constant(0.)
        done_true, done_false = tf.constant(1.), tf.constant(0.)
        policy_logits_zeros = tf.constant([0., 0., 0., 0.])
        rewards_saver = [None, None, None, None]
        obs_zeros = (tf.zeros(self._feature_maps_shape, dtype=tf.uint8),
                     tf.zeros(self._scalar_features_shape, dtype=tf.uint8))

        obsns = self._train_env.reset()
        obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)
        for i, writer in enumerate(writers):
            obs = obsns[i][0], obsns[i][1]
            obs_records.append(obs)
            if epsilon is None:
                writer.append((action_negative, policy_logits_zeros, obs, reward_zero, done_false))
            else:
                writer.append((action_negative, obs, reward_zero, done_false))
        step_counter = 1  # start with 1, since we have at least initialization
        steps_per_worker_counter = [1 for _ in range(self._n_players)]

        while True:
            if all(dones):
                steps_to_save = [None for _ in range(self._n_players)]
                # progress of an entire episode
                progress = tf.concat([tf.constant([0.]), tf.linspace(0., 1., step_counter)[:-1]], axis=0)
                if step_counter < self._n_points:
                    progress = tf.concat([progress, tf.ones(self._n_points - step_counter)], axis=0)

                for i, writer in enumerate(writers):
                    # if episode has ended before number of points required to store, put some more points
                    if step_counter < self._n_points:
                        for _ in range(self._n_points - step_counter):
                            if epsilon is None:
                                writer.append((action_negative, policy_logits_zeros, obs_zeros,
                                               rewards_saver[i], done_true))
                            else:
                                writer.append((action_negative, obs_zeros, rewards_saver[i], done_true))
                        steps_to_save[i] = self._n_points
                    # add some points which otherwise would never be initial points, maybe it can improve performance
                    elif steps_per_worker_counter[i] > (200 - self._n_points):
                        # for example if n_points = 5 and 196 is a final point, so the last trajectory saved
                        # is [192, 193, 194, 195, 196]
                        # 196 > 200 - 5 = 195 : +1 point to store [196, 197, 198, 199, 200*] or from 195?
                        # all * points (# 200* and above) are dones
                        # additional_points_to_save = step_counter - (200 - self._n_points)
                        # or just save all points left in the current writer
                        additional_points_to_save = self._n_points - 2
                        steps_to_save[i] = steps_per_worker_counter[i] + additional_points_to_save
                    else:
                        steps_to_save[i] = max(steps_per_worker_counter[i], self._n_points)

                for i, ray_writer in enumerate(ray_writers):
                    for step in range(steps_to_save[i]):
                        try:
                            action, logits, obs, reward, done = (writers[i][step][0], writers[i][step][1],
                                                                 writers[i][step][2], writers[i][step][3],
                                                                 writers[i][step][4])
                            ray_writer.append((action, logits, obs, reward, done,
                                               rewards_saver[i], progress[step]))
                            if step >= self._n_points - 1:
                                ray_writer.create_item(table=self._table_names[0],
                                                       num_timesteps=self._n_points, priority=1.)
                                if done:
                                    # ray_writer.close()
                                    if step == (steps_to_save[i] - 1):
                                        ray_writer.close()
                                    else:
                                        continue
                        except RuntimeError:
                            # continue writing with a next writer if a current one is closed
                            continue
                        except IndexError:
                            # include additional probably unreachable points
                            action, logits, obs, reward, done = (action_negative, policy_logits_zeros, obs_zeros,
                                                                 reward, done)
                            # probably this block is not necessary
                            # try:
                            #     current_progress = progress[step]
                            # except BaseException:  # progress is a tensor, it returns a base exception
                            #     current_progress = tf.constant(1.)
                            current_progress = tf.constant(1.)

                            ray_writer.append((action, logits, obs, reward, done,
                                               rewards_saver[i], current_progress))
                            ray_writer.create_item(table=self._table_names[0],
                                                   num_timesteps=self._n_points, priority=1.)
                            if step == (steps_to_save[i] - 1):
                                ray_writer.close()

                # [ray_writer.close() for ray_writer in ray_writers]
                break

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
            step_counter += 1
            actions = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.int32), actions)
            rewards = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), rewards)
            dones = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), dones)
            obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)
            for i, writer in enumerate(writers):
                action, reward, done = actions[i], rewards[i], dones[i]
                if done:
                    # the first 'done' encounter, save a final reward
                    if rewards_saver[i] is None:
                        rewards_saver[i] = reward
                        steps_per_worker_counter[i] += 1
                    # consequent 'done' encounters, put zero actions and logits
                    else:
                        action = action_negative
                        if epsilon is None:
                            policy_logits[i] = policy_logits_zeros
                    obs = obs_zeros
                    # if 'done', store final rewards
                    reward = rewards_saver[i]
                else:
                    obs = obsns[i][0], obsns[i][1]
                    steps_per_worker_counter[i] += 1
                obs_records.append(obs)
                if epsilon is None:
                    writer.append((action, policy_logits[i], obs, reward, done))
                else:
                    writer.append((action, obs, reward, done))  # returns Runtime Error if a writer is closed

        # for writer in writers:
        #     if epsilon is None:
        #         actions = tf.stack([item[0] for item in writer])
        #         policy_logits = tf.stack([item[1] for item in writer])
        #         obsns = [tf.stack([item[2][i] for item in writer]) for i in range(2)]
        #         rewards = tf.stack([item[3] for item in writer])
        #         dones = tf.stack([item[4] for item in writer])
        #         episode = (actions, policy_logits, obsns, rewards, dones)
        #     else:
        #         actions = tf.stack([item[0] for item in writer])
        #         obsns = [tf.stack([item[1][i] for item in writer]) for i in range(2)]
        #         rewards = tf.stack([item[2] for item in writer])
        #         dones = tf.stack([item[3] for item in writer])
        #         episode = (actions, obsns, rewards, dones)

        #     self._replay_memory_client.insert(episode, {self._table_names[0]: 1.})

    def _collect_some_trajectories_from_episode(self, epsilon):
        """
        Collects some trajectories (items) to a buffer. For example, if we have 4 points: 1, 2, 3, 4
        the function will store 3 trajectories to 3 'dm-reverb' tables if the 4th point is terminal:
        1, 2, 3, 4;
        2, 3, 4;
        3, 4;
        For all other point it will collect 1, 2, 3, 4 trajectory only.
        For this case there will be 3 tables, first two - small ones (for 3, 4; 2, 3, 4; type trajectories),
        the last one will be the largest (for the 1, 2, 3, 4; type trajectories)

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

        for step in it.count(2):  # the first initial step was sampled
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

                    if step >= self._n_points:
                        # add different step items only once after an episode finishes
                        if done:
                            for steps in range(2, self._n_points + 1):
                                try:
                                    writer.create_item(table=self._table_names[steps - 2],
                                                       num_timesteps=steps, priority=1.)
                                except ValueError:
                                    # stop new items creation if there are not enough buffered timesteps
                                    break
                        # add the largest step item each step
                        else:
                            try:
                                writer.create_item(table=self._table_names[-1],
                                                   num_timesteps=self._n_points, priority=1.)
                            except ValueError:
                                break
                    if done:
                        writer.close()
                except RuntimeError:
                    # continue writing with a next writer if a current one is closed
                    continue
            if all(dones):
                break

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

    def _collect_several_episodes(self, n_episodes, epsilon=None, is_random=False):
        for i in range(n_episodes):
            self._collect(epsilon, is_random)

    # def _collect_until_items_created(self, n_items, epsilon=None):
    #     # collect more exp if we do not have enough for a batch
    #     items_created = sum([item.current_size for item in self._replay_memory_client.server_info().values()])
    #     while items_created < n_items:
    #         self._collect(epsilon)
    #         items_created = sum([item.current_size for item in self._replay_memory_client.server_info().values()])

    def _sample_experience(self, fraction=None):
        samples = []
        if self._is_full_episode:
            iterator = self._iterators[0]
            samples.append(next(iterator))
            self._items_sampled[0] += self._sample_batch_size
        elif self._is_all_trajectories:
            for i, iterator in enumerate(self._iterators):
                trigger = tf.random.uniform(shape=[])
                # sample 2 steps all the time, further steps with decreasing probability no less than 0.25
                if i > 0 and trigger > max(1. / (i + 1.), 0.25):
                    samples.append(None)
                else:
                    samples.append(next(iterator))
                    self._items_sampled[i] += self._sample_batch_size
        else:
            # sampling for _collect_some_trajectories_
            for i, iterator in enumerate(self._iterators):
                quota = fraction[i] / fraction[-1]
                # train 5 times more often in all tables except the last one
                if quota < 5:
                    # if i == 1:
                    #     print(self._sampling_meter)
                    #     self._sampling_meter = 0
                    samples.append(next(iterator))
                    self._items_sampled[i] += self._sample_batch_size
                else:
                    # if i == 1:
                    #     self._sampling_meter += 1
                    samples.append(None)

        return samples

    def _prepare_td_arguments(self, actions, observations, rewards, dones, steps):
        exponents = tf.expand_dims(tf.range(steps - 1, dtype=tf.float32), axis=1)
        gammas = tf.fill([steps - 1, 1], self._discount_rate.numpy())
        discounted_gammas = tf.pow(gammas, exponents)

        total_rewards = tf.squeeze(tf.matmul(rewards[:, 1:], discounted_gammas))
        first_observations = tf.nest.map_structure(lambda x: x[:, 0, ...], observations)
        last_observations = tf.nest.map_structure(lambda x: x[:, -1, ...], observations)
        last_dones = dones[:, -1]
        last_discounted_gamma = self._discount_rate ** (steps - 1)
        second_actions = actions[:, 1]
        return total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions

    def _get_learning_rate(self, data_cnt, batch_cnt, steps):
        self._data_cnt_ema = self._data_cnt_ema * 0.8 + data_cnt / (1e-2 + batch_cnt) * 0.2
        # print(f"Data count EMA: {self._data_cnt_ema}")
        lr = self._default_lr * self._data_cnt_ema / (1 + steps * 1e-5)
        return lr

    def _training_step(self, *args, **kwargs):
        raise NotImplementedError

    def _training_step_full(self, *args, **kwargs):
        raise NotImplementedError

    def _train(self, samples_in):
        for i, sample in enumerate(samples_in):
            if sample is not None:
                # passing i to tf function as python variable cause the same number of retraces
                # as passing it as a tf constant
                # i = tf.constant(i, dtype=tf.float32)
                # also passing a tuple of tf constants does not cause retracing
                if self._is_policy_gradient:
                    if self._is_full_episode:
                        data_count = self._training_step_full(*sample.data, self._n_points, sample.info)
                        return data_count
                    else:
                        self._training_step(*sample.data, i + 2, info=info)
                else:
                    action, obs, reward, done = sample.data
                    key, probability, table_size, priority = sample.info
                    experiences, info = (action, obs, reward, done), (key, probability, table_size, priority)
                    if self._is_full_episode:
                        self._training_step_full(*experiences, steps=self._n_points, info=info)
                    else:
                        self._training_step(*experiences, steps=i + 2, info=info)

    def do_train(self, iterations_number=20000, save_interval=2000):

        target_model_update_interval = 3000
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self._start_epsilon,
            decay_steps=iterations_number,
            end_learning_rate=self._final_epsilon) if self._start_epsilon is not None else None

        weights = None
        mask = None
        # rewards = 0
        # steps = 0
        print_interval = 100
        update_interval = print_interval
        # eval_counter = 0
        data_counter = 0

        lr = self._default_lr * self._data_cnt_ema
        self._optimizer.learning_rate.assign(lr)

        # wait if there are not enough data in the buffer
        while True:
            items_created = []
            for table_name in self._table_names:
                server_info = self._replay_memory_client.server_info()[table_name]
                items_total = server_info.current_size + server_info.num_deleted_episodes
                items_created.append(items_total)

            if items_created[-1] < self._sample_batch_size:
                print("Waiting to collect enough data.")
                time.sleep(1)
                continue
            else:
                break

        weights = self._model.get_weights()
        # print(f" Variables: {len(self._model.trainable_variables)}")
        ray.get(self._workers_info.set_current_weights.remote((weights, 0)))

        # the main training loop
        for step_counter in range(1, iterations_number + 1):

            # sampling
            samples = self._sample_experience()

            # training
            # t1 = time.time()
            data_count = self._train(samples)
            data_counter += data_count.numpy()
            # t2 = time.time()
            # print(f"Training. Step: {step_counter} Time: {t2 - t1}")

            if step_counter % update_interval == 0:
                if not self._ray_queue.full():
                    weights = self._model.get_weights()
                    self._ray_queue.put(weights)  # send weights to the interprocess ray queue
                    # print("Put weights in queue.")

            if step_counter % print_interval == 0:
                lr = self._get_learning_rate(data_counter, print_interval, step_counter)
                self._optimizer.learning_rate.assign(lr)
                # lr = self._optimizer.learning_rate.numpy()
                data_counter = 0

                items_prev = items_created
                # get from a buffer the total number of created elements since a buffer initialization
                items_created = []
                for table_name in self._table_names:
                    server_info = self._replay_memory_client.server_info()[table_name]
                    items_total = server_info.current_size + server_info.num_deleted_episodes
                    items_created.append(items_total)

                # fraction = [x / y if x != 0 else 1.e-9 for x, y in zip(self._items_sampled, items_created)]
                per_step_items_created = items_created[-1] - items_prev[-1]
                if per_step_items_created == 0:
                    step_fraction = self._sample_batch_size * print_interval
                else:
                    step_fraction = self._sample_batch_size * print_interval / per_step_items_created

                print(f"Step: {step_counter}, Sampled: {self._items_sampled[0]}, "
                      f"Created total: {items_created[0]}, "
                      f"Step sample/creation frac: {step_fraction:.2f}, "
                      f"LR: {lr:.2e}")

            # evaluation
            if step_counter % save_interval == 0:
                # eval_counter += 1
                # epsilon = 0 if epsilon_fn is not None else None
                # mean_episode_reward, mean_steps = self._evaluate_episodes(epsilon=epsilon)
                # print("----Evaluation------------------")
                # print(f"Iteration:{step_counter:.2f}; "
                #       f"Reward: {mean_episode_reward:.2f}; "
                #       f"Steps: {mean_steps:.2f}")
                # print("--------------------------------")
                # rewards += mean_episode_reward
                # steps += mean_steps

                weights = self._model.get_weights()
                ray.get(self._workers_info.set_current_weights.remote((weights, step_counter)))
                data = {
                    'weights': weights,
                }
                with open(f'data/data{step_counter}.pickle', 'wb') as f:
                    pickle.dump(data, f, protocol=4)

                # with open('data/checkpoint', 'w') as text_file:
                #     checkpoint = self._replay_memory_client.checkpoint()
                #     print(checkpoint, file=text_file)

            # update target model weights
            if self._target_model and step_counter % target_model_update_interval == 0:
                weights = self._model.get_weights()
                self._target_model.set_weights(weights)

            # store weights at the last step
            if step_counter % iterations_number == 0:
                # print("----Final-results---------------")
                # epsilon = 0 if epsilon_fn is not None else None
                # mean_episode_reward, mean_steps = self._evaluate_episodes(num_episodes=10, epsilon=epsilon)
                # print(f"Final reward with a model policy is {mean_episode_reward:.2f}; "
                #       f"Final average steps survived is {mean_steps:.2f}")
                # output_reward = rewards / eval_counter
                # output_steps = steps / eval_counter
                # print(f"Average episode reward with a model policy is {output_reward:.2f}; "
                #       f"Final average per episode steps survived is {output_steps:.2f}")
                # print("--------------------------------")

                weights = self._model.get_weights()
                mask = list(map(lambda x: np.where(np.abs(x) < 0.1, 0., 1.), weights))

                if self._make_checkpoint:
                    try:
                        checkpoint = self._replay_memory_client.checkpoint()
                    except RuntimeError as err:
                        print(err)
                        checkpoint = err
                else:
                    checkpoint = None

                # disable collectors
                if self._workers_info is not None:
                    ray.get(self._workers_info.set_done.remote(True))

                break

        return weights, mask, checkpoint

    def do_train_collect(self, iterations_number=20000, eval_interval=2000):

        target_model_update_interval = 3000
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self._start_epsilon,
            decay_steps=iterations_number,
            end_learning_rate=self._final_epsilon) if self._start_epsilon is not None else None

        weights = None
        mask = None
        rewards = 0
        steps = 0
        eval_counter = 0
        print_interval = 10
        data_counter = 0
        is_random = True

        items_created = []
        for table_name in self._table_names:
            server_info = self._replay_memory_client.server_info()[table_name]
            items_total = server_info.current_size + server_info.num_deleted_episodes
            items_created.append(items_total)

        lr = self._default_lr * self._data_cnt_ema
        self._optimizer.learning_rate.assign(lr)

        for step_counter in range(1, iterations_number + 1):
            # collecting
            # items_created = [self._replay_memory_client.server_info()[table_name].current_size
            #                  for table_name in self._table_names]
            # do not collect new experience if we have not used previous
            fraction = [x / y if (x != 0 and y != 0) else None for x, y in zip(self._items_sampled, items_created)]

            # sampling
            samples = self._sample_experience(fraction)

            # training
            # t1 = time.time()
            data_count = self._train(samples)
            data_counter += data_count.numpy()
            # t2 = time.time()
            # print(f"Training. Step: {step_counter} Time: {t2 - t1}")

            # sample items (and train) 10 times more than collecting items to the last table
            # if fraction[-1] > 10:
            epsilon = epsilon_fn(step_counter) if epsilon_fn is not None else None
            # t1 = time.time()
            if step_counter > 10:
                is_random = False
            self._collect(epsilon, is_random)
            # t2 = time.time()
            # print(f"Collecting. Step: {step_counter} Time: {t2-t1}")

            if step_counter % print_interval == 0:
                lr = self._get_learning_rate(data_counter, print_interval, step_counter)
                self._optimizer.learning_rate.assign(lr)
                data_counter = 0

                items_prev = items_created
                items_created = []
                for table_name in self._table_names:
                    server_info = self._replay_memory_client.server_info()[table_name]
                    items_total = server_info.current_size + server_info.num_deleted_episodes
                    items_created.append(items_total)

                per_step_items_created = items_created[-1] - items_prev[-1]
                if per_step_items_created == 0:
                    step_fraction = self._sample_batch_size * print_interval
                else:
                    step_fraction = self._sample_batch_size * print_interval / per_step_items_created

                print(f"Step: {step_counter}, Sampled current epoch: {self._items_sampled[0]}, "
                      f"Created total: {items_created[0]}, "
                      f"Sample / creation frac: {step_fraction:.2f}, "
                      f"Learning rate: {lr:.2e}, "
                      f"Data cnt ema: {self._data_cnt_ema:.2f}")

            # evaluation
            if step_counter % eval_interval == 0:
                eval_counter += 1
                epsilon = 0 if epsilon_fn is not None else None
                mean_episode_reward, mean_steps = self._evaluate_episodes(epsilon=epsilon)
                print(f"Iteration:{step_counter:.2f}; "
                      f"Reward: {mean_episode_reward:.2f}; "
                      f"Steps: {mean_steps:.2f}")
                rewards += mean_episode_reward
                steps += mean_steps

            # update target model weights
            if self._target_model and step_counter % target_model_update_interval == 0:
                weights = self._model.get_weights()
                self._target_model.set_weights(weights)

            # store weights at the last step
            if step_counter % iterations_number == 0:
                epsilon = 0 if epsilon_fn is not None else None
                mean_episode_reward, mean_steps = self._evaluate_episodes(num_episodes=10, epsilon=epsilon)
                print(f"Final reward with a model policy is {mean_episode_reward:.2f}; "
                      f"Final average steps survived is {mean_steps:.2f}")
                output_reward = rewards / eval_counter
                output_steps = steps / eval_counter
                print(f"Average episode reward with a model policy is {output_reward:.2f}; "
                      f"Final average per episode steps survived is {output_steps:.2f}")

                weights = self._model.get_weights()
                mask = list(map(lambda x: np.where(np.abs(x) < 0.1, 0., 1.), weights))

                if self._make_checkpoint:
                    try:
                        checkpoint = self._replay_memory_client.checkpoint()
                    except RuntimeError as err:
                        print(err)
                        checkpoint = err
                else:
                    checkpoint = None

        return weights, mask, output_reward, output_steps, checkpoint
