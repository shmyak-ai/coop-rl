import random
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import ray

from coop_rl.members import (
    Member,
    Agent
)


class DQNAgent(Agent):

    def __init__(self, run_config, *args, **kwargs):
        super().__init__(run_config, *args, **kwargs)

        self._target_model = Member.member_config['model'][run_config.model](
            run_config.n_features,
            run_config.n_layers,
            run_config.seed,
        )
        self._target_model.set_weights(self._model.get_weights())

    @tf.function
    def _predict(self, observation):
        return self._model(observation)

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

    def _training_step(self, actions, observations, rewards, dones, steps, info):
        print("Tracing")
        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones, steps)

        next_Q_values = self._model(last_observations)
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self._n_outputs, dtype=tf.float32)
        next_best_Q_values = tf.reduce_sum((self._target_model(last_observations) * next_mask), axis=1)

        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * next_best_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(first_observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))  # todo: reduce_mean is redundant
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    def train(self):
        """
        Runs a step of training.
        """
        step = next(self.do_train)
        if step % self._run_config.print_interval == 0:
            print(f"Iteration number {step}.")
        if step % self._run_config.save_interval == 0:
            data = {
                'weights': self._model.get_weights(),
            }
            with open(f'data/data{step:05d}.pickle', 'wb') as f:
                pickle.dump(data, f, protocol=4)
        if step % self._run_config.weights_update_interval == 0:
            ray.get(self._exchange_actor.set_current_weights.remote((self._model.get_weights(), 0)))
        if step % self._run_config.target_model_update_interval == 0:
                self._target_model.set_weights(self._model.get_weights)

    def training(self):
        """
        Trains a sequence of steps.
        """
        while True:
            try:
                self._sample_experience()
            except ValueError:
                continue
            break
        self.train()


class PercDQNAgent(DQNAgent):

    def _training_step(self, actions, observations, rewards, dones, steps, info):
        print("Tracing")
        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones, steps)

        next_Q_values = self._target_model(last_observations)
        next_best_Q_values = tfp.stats.percentile(next_Q_values, q=99., interpolation='linear', axis=1)

        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * next_best_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(first_observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
