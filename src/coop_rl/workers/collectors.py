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

import elements
import jax
import numpy as np

from coop_rl.base.base_types import TimeStepDQN
from coop_rl.workers.auxiliary import CommandExecutor


class CollectorDQNUniform:
    def __init__(
        self,
        *,
        controller,
        trainer,
        collectors_seed,
        log_level,
        report_period,
        state_recover,
        args_state_recover,
        env,
        args_env,
        time_step_dtypes,
        get_select_action_fn,
        args_get_select_action_fn,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.report_period = report_period

        self.controller = controller
        self.trainer = trainer
        self.command_executor = CommandExecutor(max_workers=1)

        self.env = env(**args_env)

        self.dtypes = time_step_dtypes()

        self.collector_seed = collectors_seed
        self._random = random.Random(collectors_seed)
        self._rng = jax.random.PRNGKey(collectors_seed)
        self._rng, rng = jax.random.split(self._rng)
        args_state_recover.rng = rng
        flax_state = state_recover(**args_state_recover)

        # online params are to prevent dqn algs from freezing
        self.online_params = deque(maxlen=10)
        self.online_params.append(flax_state.params)

        self.futures_parameters = self.command_executor.submit(self.controller, "get_parameters")

        args_get_select_action_fn.apply_fn = flax_state.apply_fn 
        self.select_action = get_select_action_fn(**args_get_select_action_fn)
        self.obs = None
        self.episode_reward = {
            "now": 0,
            "last": 0,
        }

    def run_rollout(self) -> list[TimeStepDQN]:
        trajectory_steps: list[TimeStepDQN] = []

        for _ in range(100):
            self._rng, action_jnp = self.select_action(
                self._rng,
                self._random.choice(self.online_params),
                self.obs,
            )
            action_np = np.asarray(action_jnp, dtype=self.dtypes.action).squeeze()
            next_obs, reward, terminated, truncated, _info = self.env.step(action_np)

            trajectory_steps.append(
                TimeStepDQN(
                    obs=np.asarray(self.obs, dtype=self.dtypes.obs),
                    action=np.asarray(action_np, dtype=self.dtypes.action),
                    reward=np.asarray(reward, dtype=self.dtypes.reward),
                    terminated=np.asarray(terminated, dtype=self.dtypes.terminated),
                    truncated=np.asarray(truncated, dtype=self.dtypes.truncated),
                )
            )
            self.episode_reward["now"] += reward

            if terminated or truncated:
                next_obs, _info = self.env.reset()
                self.episode_reward["last"] = self.episode_reward["now"]
                self.episode_reward["now"] = 0

            self.obs = next_obs

        return trajectory_steps

    def collecting(self):
        try:
            self._collecting()
        finally:
            self.close()

    def _collecting(self):
        self.obs, _ = self.env.reset()
        for rollouts_count in itertools.count(start=1, step=1):
            trajectory_steps = self.run_rollout()
            trajectory = TimeStepDQN(
                obs=np.stack([step.obs for step in trajectory_steps], axis=0),
                action=np.asarray(
                    [step.action for step in trajectory_steps], dtype=self.dtypes.action
                ),
                reward=np.asarray(
                    [step.reward for step in trajectory_steps], dtype=self.dtypes.reward
                ),
                terminated=np.asarray(
                    [step.terminated for step in trajectory_steps], dtype=self.dtypes.terminated
                ),
                truncated=np.asarray(
                    [step.truncated for step in trajectory_steps], dtype=self.dtypes.truncated
                ),
            )

            while True:
                training_done = self.command_executor.call(self.controller, "is_done")
                if training_done:
                    self.logger.info("Done signal received; finishing.")
                    return

                adding_traj_done = self.command_executor.call(
                    self.trainer,
                    "add_traj_seq",
                    (
                        self.collector_seed,
                        trajectory,
                    ),
                )
                if adding_traj_done:
                    break
                time.sleep(0.1)

            parameters = self.command_executor.resolve(self.futures_parameters)
            if parameters is not None:
                self.online_params.append(parameters)
            self.futures_parameters = self.command_executor.submit(
                self.controller,
                "get_parameters",
            )

            if rollouts_count % self.report_period == 0:
                self.logger.info(f"Last episode reward: {self.episode_reward['last']:.4f}.")

    def close(self) -> None:
        """Release local helper resources after collection stops."""
        self.command_executor.shutdown()
        self.env.close()


class CollectorDreamerUniform:
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
        get_select_action_fn,
        controller,
        trainer,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.report_period = report_period

        self.controller = controller
        self.trainer = trainer
        self.command_executor = CommandExecutor()

        self.env = env(**args_env)

        self.collector_seed = collectors_seed
        random.seed(collectors_seed)
        args_state_recover["rng"] = jax.random.PRNGKey(collectors_seed)
        self.flax_state = state_recover(**args_state_recover)

        self.futures_parameters = self.command_executor.submit(self.controller, "get_parameters")
        self.select_action = get_select_action_fn(self.flax_state)
        self.action = {
            k: np.zeros((1,) + v.shape, v.dtype) for k, v in self.env._env.act_space.items()
        }
        self.action["reset"] = np.ones(1, bool)
        self.episode_reward = {
            "now": 0,
            "last": 0,
        }
        self.gpu_device = jax.devices("gpu")[0]
        self.rollout_length = 1000

    def run_rollout(self):
        trajectory = []
        uuid = elements.UUID()
        for index in range(self.rollout_length):
            action = {k: v[0] for k, v in self.action.items()}
            obs = self.env.step(action)
            obs = {
                k: np.stack(
                    [
                        obs[k],
                    ]
                )
                for k in obs
            }
            obs = {k: v for k, v in obs.items() if not k.startswith("log/")}
            self.flax_state, self.action, outs = self.select_action(self.flax_state, obs)
            self.action = {**self.action, "reset": obs["is_last"].copy()}

            step_id = np.expand_dims(
                np.frombuffer(bytes(uuid) + index.to_bytes(4, "big"), np.uint8), axis=0
            )
            trajectory.append(
                {
                    "image": obs["image"],
                    "is_first": obs["is_first"],
                    "is_last": obs["is_last"],
                    "is_terminal": obs["is_terminal"],
                    "reward": obs["reward"],
                    "stepid": step_id,
                    "dyn/deter": outs["dyn/deter"],
                    "dyn/stoch": outs["dyn/stoch"],
                    "action": self.action["action"],
                }
            )
            self.episode_reward["now"] += obs["reward"][0]

            if obs["is_terminal"][0] or obs["is_last"][0]:
                self.episode_reward["last"] = self.episode_reward["now"]
                self.episode_reward["now"] = 0

        trajectory = {k: np.concatenate([x[k] for x in trajectory], axis=0) for k in trajectory[0]}
        trajectory["consec"] = np.full(trajectory["is_first"].shape, 0, np.int32)
        return trajectory

    def collecting(self):
        for rollouts_count in itertools.count(start=1, step=1):
            trajectory = self.run_rollout()

            while True:
                training_done = self.command_executor.call(self.controller, "is_done")
                if training_done:
                    self.logger.info("Done signal received; finishing.")
                    return

                adding_traj_done = self.command_executor.call(
                    self.trainer,
                    "add_traj_seq",
                    (
                        self.collector_seed,
                        trajectory,
                    ),
                )
                if adding_traj_done:
                    break
                time.sleep(0.1)

            parameters = self.command_executor.resolve(self.futures_parameters)
            if parameters is not None:
                self.flax_state = self.flax_state.update_state(
                    jax.device_put(parameters, device=self.gpu_device),
                    self.flax_state.carry,
                    self.flax_state.carry_train,
                )
            self.futures_parameters = self.command_executor.submit(
                self.controller,
                "get_parameters",
            )

            if rollouts_count % self.report_period == 0:
                self.logger.info(f"Last episode reward: {self.episode_reward['last']:.4f}.")

    def close(self) -> None:
        """Release local helper resources after collection stops."""
        self.command_executor.shutdown()
        self.env.close()
