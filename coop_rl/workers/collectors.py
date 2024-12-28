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
import random
import time
from collections import deque

import jax
import numpy as np
import ray

from coop_rl.agents.dqn import select_action


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
        epsilon_fn,
        epsilon=0.01,
        epsilon_decay_period=250000,
        flax_state,
        env,
        args_env,
        controller,
        trainer,
        preprocess_fn,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.report_period = report_period

        self.controller = controller
        self.trainer = trainer

        self.env = env(**args_env)

        self.num_actions = num_actions
        self.network = network(**args_network)
        self.preprocess_fn = preprocess_fn

        self.collector_seed = collectors_seed
        random.seed(collectors_seed)
        self._rng = jax.random.PRNGKey(collectors_seed)
        # to improve obs diversity during exp collection
        self.online_params = deque(maxlen=10)
        if flax_state is None:
            self._build_network(observation_shape)
        else:
            # network.init gives a dict "params"
            # network.apply also needs "params"
            self.online_params.append({"params": flax_state.params})

        self.epsilon_fn = epsilon_fn
        self.epsilon = epsilon
        self.epsilon_decay_period = epsilon_decay_period
        self.epsilon_current = None

        self.obs = None
        self.collecting_steps = 0
        self.warmup_steps = warmup_steps

        parameters = ray.get(self.controller.get_parameters.remote())
        self.futures_parameters = self.controller.get_parameters.remote()
        if parameters is not None:
            self.online_params.append(parameters)

    def _build_network(self, observation_shape):
        self._rng, rng = jax.random.split(self._rng)
        state = self.preprocess_fn(np.ones((1, *observation_shape)))
        self.online_params.append(self.network.init(rng, x=state))

    def run_rollout(self):
        traj_obs = []
        traj_actions = []
        traj_rewards = []
        traj_terminated = []
        traj_truncated = []

        if self.obs is None:
            self.obs, _info = self.env.reset()
        for _ in range(100):
            self._rng, action, self.epsilon_current = select_action(
                self.network,
                random.choice(self.online_params),
                self.preprocess_fn(self.obs),
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
            action_np = np.asarray(action)
            next_obs, reward, terminated, truncated, _info = self.env.step(action_np)

            traj_obs.append(self.obs)
            traj_actions.append(action_np)
            traj_rewards.append(reward)
            traj_terminated.append(terminated)
            traj_truncated.append(truncated)

            if terminated or truncated:
                next_obs, _info = self.env.reset()

            self.obs = next_obs

        return traj_obs, traj_actions, traj_rewards, traj_terminated, traj_truncated

    def collecting(self):
        episodes_rewards = []
        for episodes_count in itertools.count(start=1, step=1):
            traj_obs, traj_actions, traj_rewards, traj_terminated, traj_truncated = self.run_rollout()
            traj_obs_np = np.array(traj_obs, dtype=np.float32)
            traj_actions_np = np.array(traj_actions, dtype=np.int32)
            traj_rewards_np = np.array(traj_rewards, dtype=np.float32)
            traj_terminated_np = np.array(traj_terminated, dtype=np.int32)
            traj_truncated_np = np.array(traj_truncated, dtype=np.int32)

            while True:
                training_done = ray.get(self.controller.is_done.remote())
                if training_done:
                    self.logger.info("Done signal received; finishing.")
                    return

                adding_traj_done = ray.get(
                    self.trainer.add_traj_seq.remote(
                        (
                            self.collector_seed,
                            traj_obs_np,
                            traj_actions_np,
                            traj_rewards_np,
                            traj_terminated_np,
                            traj_truncated_np,
                        )
                    )
                )
                if adding_traj_done:
                    break
                time.sleep(0.1)

            parameters = ray.get(self.futures_parameters)
            if parameters is not None:
                self.online_params.append(parameters)
            self.futures_parameters = self.controller.get_parameters.remote()

            episodes_rewards.append(sum(traj_rewards))
            if episodes_count % self.report_period == 0:
                self.logger.info(f"Mean episode reward: {sum(episodes_rewards) / len(episodes_rewards):.4f}.")
                self.logger.debug(f"Current epsilon: {float(self.epsilon_current)}.")
                self.logger.debug(f"Online params deque size: {len(self.online_params)}.")
                episodes_rewards = []
