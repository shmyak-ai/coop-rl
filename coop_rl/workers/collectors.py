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

import itertools
import logging
import time

import jax
import numpy as np
import ray

from coop_rl import networks
from coop_rl.utils import (
    linearly_decaying_epsilon,
    select_action,
)


class DQNCollectorUniform:
    def __init__(
        self,
        report_period,
        num_actions,
        observation_shape,
        network,
        handler_env,
        args_handler_env,
        handler_replay,
        args_handler_replay,
        collector_id=0,
        warmup_steps=10000,
        epsilon_fn=linearly_decaying_epsilon,
        epsilon=0.01,
        epsilon_decay_period=250000,
        control_actor=None,
        replay_actor=None,
        seed=None,
        preprocess_fn=None,
    ):
        self.logger  = logging.getLogger("ray")
        self.report_period = report_period

        self.control_actor = control_actor
        self.replay_actor = replay_actor

        assert isinstance(observation_shape, tuple)
        seed = int(time.time() * 1e6) if seed is None else seed

        self.num_actions = num_actions

        self._environment = handler_env(**args_handler_env)

        if preprocess_fn is None:
            self.network = network(num_actions=num_actions)
            self.preprocess_fn = networks.identity_preprocess_fn
        else:
            self.network = network(num_actions=num_actions, inputs_preprocessed=True)
            self.preprocess_fn = preprocess_fn

        self.epsilon_fn = epsilon_fn
        self.epsilon = epsilon
        self.epsilon_decay_period = epsilon_decay_period
        self.epsilon_current = None
        self.collecting_steps = 0
        self.warmup_steps = warmup_steps

        self._replay = handler_replay(**args_handler_replay)  # to store episode transitions

        self.logger.debug(f"Seed: {seed + collector_id}.")
        self._rng = jax.random.key(seed + collector_id)

        self._observation = np.zeros(observation_shape)
        self._build_network()

    def _build_network(self):
        self._rng, rng = jax.random.split(self._rng)
        state = self.preprocess_fn(self._observation)
        self.online_params = self.network.init(rng, x=state)

    def _initialize_episode(self):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """

        self._rng, rng = jax.random.split(self._rng)
        seed = jax.random.bits(rng)
        self._observation, info = self._environment.reset(seed=int(seed))

        self._rng, action, self.epsilon_current = select_action(
            self.network,
            self.online_params,
            self.preprocess_fn(self._observation),
            self._rng,
            self.num_actions,
            False,  # eval mode
            0.001,  # epsilon_eval,
            self.epsilon,  # epsilon_train,
            self.epsilon_decay_period,
            self.collecting_steps,
            self.warmup_steps,
            self.epsilon_fn,
        )
        return np.asarray(action)

    def _step(self, action, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
          action: int, the most recent action.
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.

        Returns:
          int, the selected action.
        """
        last_observation = self._observation
        self._observation = observation

        self._replay._store_transition(last_observation, action, reward, False)

        self._rng, action, self.epsilon_current = select_action(
            self.network,
            self.online_params,
            self.preprocess_fn(self._observation),
            self._rng,
            self.num_actions,
            False,  # eval mode
            0.001,  # epsilon_eval,
            self.epsilon,  # epsilon_train,
            self.epsilon_decay_period,
            self.collecting_steps,
            self.warmup_steps,
            self.epsilon_fn,
        )
        self.collecting_steps += 1
        return np.asarray(action)

    def _end_episode(self, action, reward, terminated, truncated):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Args:
          action: int, the last action.
          reward: float, the last reward from the environment.
          episode_end: bool, whether the last state-action led to a terminal state.
        """
        self._replay._store_transition(self._observation, action, reward, terminated, truncated=truncated)

    def run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """

        steps = 0
        rewards = 0
        try:
            self._replay.reset()
            action = self._initialize_episode()

            # Keep interacting until terminated / truncated state.
            while True:
                observation, reward, terminated, truncated, info = self._environment.step(action)
                steps += 1
                rewards += reward

                if terminated or truncated:
                    break
                else:
                    action = self._step(action, reward, observation)

            self._end_episode(action, reward, terminated, truncated)
        finally:
            self._replay.close()
        return steps, rewards

    def collecting_dopamine(self, num_episodes):
        for _ in range(num_episodes):
            self.run_one_episode()
            self.replay_actor.add_episode(self._replay.replay)

    def collecting_reverb(self, num_episodes):
        for _ in range(num_episodes):
            self.run_one_episode()

    def collecting_dopamine_remote(self):
        while True:
            parameters, done = ray.get(self.control_actor.get_parameters_done.remote())
            if done:
                self.logger.info("Done signal received; finishing.")
                break
            if parameters is not None:
                self.online_params = parameters
            self.run_one_episode()
            ray.get(self.replay_actor.add_episode.remote(self._replay.replay))

    def collecting_reverb_remote(self):
        episode_steps = []
        episode_rewards = []
        for episodes_count in itertools.count(start=1, step=1):
            parameters, done = ray.get(self.control_actor.get_parameters_done.remote())
            if done:
                self.logger.info("Done signal received; finishing.")
                break
            if parameters is not None:
                self.online_params = parameters
            steps, rewards = self.run_one_episode()
            episode_steps.append(steps)
            episode_rewards.append(rewards)
            if episodes_count % self.report_period == 0:
                self.logger.info(f"Mean episode length: {sum(episode_steps) / len(episode_steps):.4f}.")
                self.logger.info(f"Mean episode reward: {sum(episode_rewards) / len(episode_rewards):.4f}.")
                self.logger.info(f"Current epsilon: {float(self.epsilon_current)}.")
                episode_steps = []
                episode_rewards = []
