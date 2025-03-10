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


class DQNCollectorUniform:
    def __init__(
        self,
        *,
        collectors_seed,
        log_level,
        report_period,
        state_recover,
        args_state_recover,
        env,
        args_env,
        neptune_run,
        args_neptune_run,
        get_select_action_fn,
        time_step_dtypes,
        controller,
        trainer,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.report_period = report_period

        self.controller = controller
        self.trainer = trainer

        self.env = env(**args_env)

        args_neptune_run["monitoring_namespace"] = f"monitoring/collector_{collectors_seed}"
        self.neptune_run = neptune_run(**args_neptune_run)
        self.collector_ns = self.neptune_run[f"collector_{collectors_seed}"]

        self.dtypes = time_step_dtypes

        self.collector_seed = collectors_seed
        random.seed(collectors_seed)
        self._rng = jax.random.PRNGKey(collectors_seed)
        self._rng, rng = jax.random.split(self._rng)
        flax_state = state_recover(rng, **args_state_recover)

        # online params are to prevent dqn algs from freezing
        self.online_params = deque(maxlen=10)
        self.online_params.append(flax_state.params)

        self.futures_parameters = self.controller.get_parameters.remote()

        self.select_action = get_select_action_fn(flax_state.apply_fn)
        self.episode_reward = {
            "now": 0,
            "last": 0,
        }

    def run_rollout(self):
        traj_obs = []
        traj_actions = []
        traj_rewards = []
        traj_terminated = []
        traj_truncated = []

        for _ in range(100):
            self._rng, action_jnp = self.select_action(
                self._rng,
                random.choice(self.online_params),
                self.obs,
            )
            action_np = np.asarray(action_jnp, dtype=self.dtypes.action).squeeze()
            next_obs, reward, terminated, truncated, _info = self.env.step(action_np)

            traj_obs.append(self.obs)
            traj_actions.append(action_np)
            traj_rewards.append(reward)
            traj_terminated.append(terminated)
            traj_truncated.append(truncated)
            self.episode_reward["now"] += reward

            if terminated or truncated:
                next_obs, _info = self.env.reset()
                self.episode_reward["last"] = self.episode_reward["now"]
                self.episode_reward["now"] = 0
                self.collector_ns["episode_reward"].append(self.episode_reward["last"])

            self.obs = next_obs

        return traj_obs, traj_actions, traj_rewards, traj_terminated, traj_truncated

    def collecting(self):
        self.obs, _ = self.env.reset()
        for rollouts_count in itertools.count(start=1, step=1):
            traj_obs, traj_actions, traj_rewards, traj_terminated, traj_truncated = self.run_rollout()
            traj_obs_np = np.array(traj_obs, dtype=self.dtypes.obs)
            traj_actions_np = np.array(traj_actions, dtype=self.dtypes.action)
            traj_rewards_np = np.array(traj_rewards, dtype=self.dtypes.reward)
            traj_terminated_np = np.array(traj_terminated, dtype=self.dtypes.terminated)
            traj_truncated_np = np.array(traj_truncated, dtype=self.dtypes.truncated)

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
                self.collector_ns["parameters_updated_on_rollout_count"].append(rollouts_count)
            self.futures_parameters = self.controller.get_parameters.remote()

            if rollouts_count % self.report_period == 0:
                self.logger.info(f"Last episode reward: {self.episode_reward['last']:.4f}.")
