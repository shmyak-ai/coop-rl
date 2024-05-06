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

import time

import gymnasium as gym
import jax
import numpy as np
import ray

from coop_rl import networks
from coop_rl.replay_memory import prioritized_replay_buffer
from coop_rl.utils import (
    identity_epsilon,
    select_action,
)


class DQNCollector:
    def __init__(
        self,
        control_actor,
        replay_actor,
        collector_id,
        num_actions,
        observation_shape,
        observation_dtype,
        environment,
        network,
        min_replay_history=20000,
        epsilon_fn=identity_epsilon,
        epsilon=0.01,
        epsilon_decay_period=250000,
        seed=None,
        preprocess_fn=None,
    ):
        self.control_actor = control_actor
        self.replay_actor = replay_actor
        self.collector_id = collector_id

        assert isinstance(observation_shape, tuple)
        self.seed = int(time.time() * 1e6) if seed is None else seed

        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype

        self._environment = gym.make(environment)

        if preprocess_fn is None:
            self.network_def = network(num_actions=num_actions)
            self.preprocess_fn = networks.identity_preprocess_fn
        else:
            self.network_def = network(num_actions=num_actions, inputs_preprocessed=True)
            self.preprocess_fn = preprocess_fn

        self.epsilon_fn = epsilon_fn
        self.epsilon = epsilon
        self.epsilon_decay_period = epsilon_decay_period
        self.training_steps = 0  # get remotely to use linear eps decay
        self.min_replay_history = min_replay_history

        self._replay = []  # to store episode transitions

        self._rng = jax.random.PRNGKey(self.seed)

        self._observation = np.zeros(observation_shape)
        self._build_network()
        # self.network_def.apply(self.online_params, np.random.rand(*state_shape))

    def _build_network(self):
        self._rng, rng = jax.random.split(self._rng)
        state = self.preprocess_fn(self._observation)
        self.online_params = self.network_def.init(rng, x=state)

    def _store_transition(self, last_observation, action, reward, is_terminal, *args, priority=None, episode_end=False):
        """Stores a transition when in training mode.

        Stores the following tuple in the replay buffer (last_observation, action,
        reward, is_terminal, priority).

        Args:
          last_observation: Last observation, type determined via observation_type
            parameter in the replay_memory constructor.
          action: An integer, the action taken.
          reward: A float, the reward.
          is_terminal: Boolean indicating if the current state is a terminal state.
          *args: Any, other items to be added to the replay buffer.
          priority: Float. Priority of sampling the transition. If None, the default
            priority will be used. If replay scheme is uniform, the default priority
            is 1. If the replay scheme is prioritized, the default priority is the
            maximum ever seen [Schaul et al., 2015].
          episode_end: bool, whether this transition is the last for the episode.
            This can be different than terminal when ending the episode because of a
            timeout, for example.
        """
        is_prioritized = isinstance(
            self._replay,
            prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer,
        )
        if is_prioritized and priority is None:
            priority = 1.0 if self._replay_scheme == "uniform" else self._replay.sum_tree.max_recorded_priority

        self._replay.append(
            (last_observation, action, reward, is_terminal, *args, {
                "priority": priority,
                "episode_end": episode_end,
                }
            )
        )

    def _initialize_episode(self):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """

        self._observation, info = self._environment.reset(seed=self.seed)

        self._rng, action = select_action(
            self.network_def,
            self.online_params,
            self.preprocess_fn(self._observation),
            self._rng,
            self.num_actions,
            False,  # eval mode
            0.001,  # epsilon_eval,
            self.epsilon,  # epsilon_train,
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
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

        self._store_transition(last_observation, action, reward, False)

        self._rng, action = select_action(
            self.network_def,
            self.online_params,
            self.preprocess_fn(self._observation),
            self._rng,
            self.num_actions,
            False,  # eval mode
            0.001,  # epsilon_eval,
            self.epsilon,  # epsilon_train,
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
            self.epsilon_fn,
        )
        return np.asarray(action)

    def _end_episode(self, action, reward, episode_end=True):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Args:
          action: int, the last action.
          reward: float, the last reward from the environment.
          episode_end: bool, whether the last state-action led to a terminal state.
        """
        self._store_transition(self._observation, action, reward, True, episode_end=episode_end)

    def run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.0

        action = self._initialize_episode()

        # Keep interacting until terminated / truncated state.
        while True:
            observation, reward, terminated, truncated, info = self._environment.step(action)

            total_reward += reward
            step_number += 1

            if terminated or truncated:
                break
            else:
                action = self._step(action, reward, observation)

        # truncated=True corresponds to episode_end=False in store_transition
        self._end_episode(action, reward, episode_end=not truncated)

        # send transitions from episode to the replay actor
        ray.get(self.replay_actor.add_episode.remote(self._replay))
        self._replay = []

        return step_number, total_reward
