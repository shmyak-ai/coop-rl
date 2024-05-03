import time
import itertools as it

import tensorflow as tf
import numpy as np
import ray


class Evaluator:

    @tf.function
    def _eval_predict(self, observation):
        return self._eval_model(observation)

    def evaluate_episode(self):
        obs_records = self._eval_env.reset()
        rewards_storage = np.zeros(self._n_players)
        for step in it.count(0):
            actions = []
            obsns = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs_records)
            for i in range(self._n_players):
                policy_logits, _ = self._predict(obsns[i]) if i < 2 else self._eval_predict(obsns[i])
                # policy_logits, _ = self._eval_predict(obsns[i]) if i < 2 else self._predict(obsns[i])
                # if i < 2:
                #     policy_logits, _ = self._predict(obsns[i])
                # else:
                #     policy_logits, _ = self._eval_predict(obsns[i])

                action = tf.random.categorical(policy_logits, num_samples=1, dtype=tf.int32)
                actions.append(action.numpy()[0][0])

            obs_records, rewards, dones, info = self._eval_env.step(actions)
            rewards_storage += np.asarray(rewards)
            if all(dones):
                break
        # winner = rewards_storage.argmax()
        winners = np.argwhere(rewards_storage == np.amax(rewards_storage))
        return winners

    def evaluate_episodes(self):
        wins = 0
        losses = 0
        draws = 0
        for _ in range(100):
            winners = self.evaluate_episode()
            if (0 in winners or 1 in winners) and 2 not in winners and 3 not in winners:
                wins += 1
            elif (2 in winners or 3 in winners) and 0 not in winners and 1 not in winners:
                losses += 1
            else:
                draws += 1
        return wins, losses, draws

    def do_evaluate(self):
        while True:
            is_done = ray.get(self._workers_info.get_done.remote())
            if is_done:
                # print("Evaluation is done.")
                time.sleep(1)  # is needed to have time to print the last 'total wins'
                return 'Done'
            while True:
                weights, step = ray.get(self._workers_info.get_current_weights.remote())
                if weights is None:
                    time.sleep(1)
                else:
                    # print(f" Variables: {len(self._model.trainable_variables)}")
                    self._model.set_weights(weights)
                    break

            wins, losses, draws = self.evaluate_episodes()
            print(f"Evaluator: Wins: {wins}; Losses: {losses}; Draws: {draws}; Model from a step: {step}.")
