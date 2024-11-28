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

import contextlib
import itertools
import logging
import random
from collections import deque

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
        *,
        collectors_seed,
        log_level="INFO",
        report_period=25,
        num_actions,
        observation_shape,
        network,
        args_network,
        warmup_steps=10000,
        epsilon_fn=linearly_decaying_epsilon,
        epsilon=0.01,
        epsilon_decay_period=250000,
        flax_state,
        buffer,
        controller,
        env,
        args_env,
        preprocess_fn=networks.identity_preprocess_fn,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.report_period = report_period

        self.controller = controller
        self.buffer = buffer

        self.env = env(**args_env)

        self.num_actions = num_actions
        self.network = network(**args_network)
        self.preprocess_fn = preprocess_fn

        self._rng = jax.random.PRNGKey(collectors_seed)
        self._observation = np.ones((1, *observation_shape))
        # to improve obs diversity during exp collection
        self.online_params = deque(maxlen=10)
        if flax_state is None:
            self._build_network()
        else:
            # network.init gives a dict "params"
            # network.apply also needs "params"
            self.online_params.append({"params": flax_state.params})

        self.epsilon_fn = epsilon_fn
        self.epsilon = epsilon
        self.epsilon_decay_period = epsilon_decay_period
        self.epsilon_current = None

        self.collecting_steps = 0
        self.warmup_steps = warmup_steps

        breakpoint()
        parameters = ray.get(self.controller.get_parameters.remote())
        self.futures_parameters = self.controller.get_parameters.remote()
        if parameters is not None:
            self.online_params.append(parameters)

    def _build_network(self):
        self._rng, rng = jax.random.split(self._rng)
        state = self.preprocess_fn(self._observation)
        self.online_params.append(self.network.init(rng, x=state))

    def _initialize_episode(self):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """

        self._observation, info = self.env.reset()

        self._rng, action, self.epsilon_current = select_action(
            self.network,
            random.choice(self.online_params),
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

        with contextlib.suppress(AttributeError):
            if (
                self.collecting_steps > self.trainer.min_replay_history
                and self.collecting_steps % self.train_period_steps == 0
            ):
                with contextlib.suppress(AttributeError):
                    replay_elements, fetch_time = self.trainer.sampler.sample_from_replay_buffer()
                with contextlib.suppress(AttributeError):
                    replay_elements = self.trainer.replay_actor.sample_from_replay_buffer()

                # train_step will put new parameters into control actor store
                self.trainer._train_step(replay_elements)
                # and fetch it back
                parameters = self.controller.get_parameters()
                if parameters is not None:
                    self.online_params.append(parameters)

            if self.collecting_steps % (self.trainer.target_update_period * self.train_period_steps) == 0:
                self.trainer.state = self.trainer.state.update_target_params()

        self._rng, action, self.epsilon_current = select_action(
            self.network,
            random.choice(self.online_params),
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

        rewards = 0
        try:
            self._replay.reset()
            action = self._initialize_episode()

            # Keep interacting until terminated / truncated state.
            for step in itertools.count(start=1, step=1):
                observation, reward, terminated, truncated, info = self.env.step(action)
                rewards += reward

                if terminated or truncated:
                    break
                else:
                    reward = np.clip(reward, -1, 1)
                    action = self._step(action, reward, observation)

                if step % 25 == 0:
                    with contextlib.suppress(AttributeError):
                        parameters = ray.get(self.futures_parameters)
                        if parameters is not None:
                            self.online_params.append(parameters)
                        self.futures_parameters = self.controller.get_parameters.remote()

            self._end_episode(action, reward, terminated, truncated)
        finally:
            self._replay.close()
        return step, rewards

    def collecting(self):
        episodes_steps = []
        episodes_rewards = []
        breakpoint()
        for episodes_count in itertools.count(start=1, step=1):
            done = ray.get(self.controller.is_done.remote())
            if done:
                self.logger.info("Done signal received; finishing.")
                break
            episode_steps, episode_rewards = self.run_one_episode()
            with contextlib.suppress(AttributeError):
                ray.get(self.buffer.add_episode.remote(self._replay.replay))
            episodes_steps.append(episode_steps)
            episodes_rewards.append(episode_rewards)
            if episodes_count % self.report_period == 0:
                self.logger.info(f"Mean episode length: {sum(episodes_steps) / len(episodes_steps):.4f}.")
                self.logger.info(f"Mean episode reward: {sum(episodes_rewards) / len(episodes_rewards):.4f}.")
                self.logger.debug(f"Current epsilon: {float(self.epsilon_current)}.")
                self.logger.debug(f"Online params deque size: {len(self.online_params)}.")
                episodes_steps = []
                episodes_rewards = []
