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

import inspect
import time

import jax
import numpy as onp
from absl import logging

from coop_rl import networks
from coop_rl.utils import select_action


class DQNCollector:
    def __init__(
        self,
        control_actor,
        replay_actor,
        num_actions,
        observation_shape,
        observation_dtype,
        stack_size,
        network,
        seed=None,
        preprocess_fn=None,
    ):
        assert isinstance(observation_shape, tuple)
        seed = int(time.time() * 1e6) if seed is None else seed

        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size

        if preprocess_fn is None:
            self.network_def = network(num_actions=num_actions)
            self.preprocess_fn = networks.identity_preprocess_fn
        else:
            self.network_def = network(num_actions=num_actions, inputs_preprocessed=True)
            self.preprocess_fn = preprocess_fn

        self._rng = jax.random.PRNGKey(seed)
        state_shape = self.observation_shape + (stack_size,)
        self.state = onp.zeros(state_shape)
        self._build_network()

        # Variables to be initialized by the agent once it interacts with the
        # environment.
        self._observation = None
        self._last_observation = None

    def _build_network(self):
        self._rng, rng = jax.random.split(self._rng)
        state = self.preprocess_fn(self.state)
        self.online_params = self.network_def.init(rng, x=state)

    def _reset_state(self):
        """Resets the agent state by filling it with zeros."""
        self.state.fill(0)

    def _record_observation(self, observation):
        """Records an observation and update state.

        Extracts a frame from the observation vector and overwrites the oldest
        frame in the state buffer.

        Args:
          observation: numpy array, an observation from the environment.
        """
        # Set current observation. We do the reshaping to handle environments
        # without frame stacking.
        self._observation = onp.reshape(observation, self.observation_shape)
        # Swap out the oldest frame with the current frame.
        self.state = onp.roll(self.state, -1, axis=-1)
        self.state[..., -1] = self._observation

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """
        self._reset_state()
        self._record_observation(observation)

        if not self.eval_mode:
            self._train_step()

        self._rng, self.action = select_action(
            self.network_def,
            self.online_params,
            self.preprocess_fn(self.state),
            self._rng,
            self.num_actions,
            self.eval_mode,
            self.epsilon_eval,
            self.epsilon_train,
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
            self.epsilon_fn,
        )
        self.action = onp.asarray(self.action)
        return self.action

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.

        Returns:
          int, the selected action.
        """
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(self._last_observation, self.action, reward, False)
            self._train_step()

        self._rng, self.action = select_action(
            self.network_def,
            self.online_params,
            self.preprocess_fn(self.state),
            self._rng,
            self.num_actions,
            self.eval_mode,
            self.epsilon_eval,
            self.epsilon_train,
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
            self.epsilon_fn,
        )
        self.action = onp.asarray(self.action)
        return self.action

    def end_episode(self, reward, terminal=True):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Args:
          reward: float, the last reward from the environment.
          terminal: bool, whether the last state-action led to a terminal state.
        """
        if not self.eval_mode:
            argspec = inspect.getfullargspec(self._store_transition)
            if "episode_end" in argspec.args or "episode_end" in argspec.kwonlyargs:
                self._store_transition(self._observation, self.action, reward, terminal, episode_end=True)
            else:
                logging.warning("_store_transition function doesn't have episode_end arg.")
                self._store_transition(self._observation, self.action, reward, terminal)

    def _initialize_episode(self):
        """Initialization for a new episode.

        Returns:
          action: int, the initial action chosen by the agent.
        """
        initial_observation = self._environment.reset()
        return self._agent.begin_episode(initial_observation)

    def _run_one_step(self, action):
        """Executes a single step in the environment.

        Args:
          action: int, the action to perform in the environment.

        Returns:
          The observation, reward, and is_terminal values returned from the
            environment.
        """
        observation, reward, is_terminal, _ = self._environment.step(action)
        return observation, reward, is_terminal

    def _end_episode(self, reward, terminal=True):
        """Finalizes an episode run.

        Args:
          reward: float, the last reward from the environment.
          terminal: bool, whether the last state-action led to a terminal state.
        """
        if isinstance(self._agent, jax_dqn_agent.JaxDQNAgent):
            self._agent.end_episode(reward, terminal)
        else:
            # TODO(joshgreaves): Add terminal signal to TF dopamine agents
            self._agent.end_episode(reward)

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.0

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal = self._run_one_step(action)

            total_reward += reward
            step_number += 1

            if self._clip_rewards:
                # Perform reward clipping.
                reward = np.clip(reward, -1, 1)

            if self._environment.game_over or step_number == self._max_steps_per_episode:
                # Stop the run loop once we reach the true end of episode.
                break
            elif is_terminal:
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._end_episode(reward, is_terminal)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation)

        self._end_episode(reward, is_terminal)

        return step_number, total_reward
