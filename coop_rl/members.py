"""
The abstract classes for the agents and workers
"""

import time

import tensorflow as tf
import ray
import reverb


class Agent:

    def __init__(self, buffer_server_port):
        self._network = None
        self._optimizer = None
        self._loss_fn = None

        self._replay_memory_client = reverb.Client(f'localhost:{buffer_server_port}')

    def _sample_experience(self):
        raise NotImplementedError

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
