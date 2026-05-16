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
        steps_per_rollout,
        get_select_action_fn,
        args_get_select_action_fn,
    ):
        self.logger = logging.getLogger(f"{__name__}.seed{collectors_seed}")
        self.logger.setLevel(log_level)
        self.report_period = report_period

        self.controller = controller
        self.trainer = trainer
        self.command_executor = CommandExecutor(max_workers=1)

        self.env = env(**args_env)
        self.num_envs = self.env.num_envs

        self.dtypes = time_step_dtypes()
        self.steps_per_rollout = steps_per_rollout

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
        self.episode_reward_now = np.zeros(self.num_envs)
        self.completed_returns: deque[float] = deque(maxlen=100)
        self._params_received = 0
        self._closed = False
        self.logger.info(
            "CollectorDQNUniform initialized (seed=%d, num_envs=%d).",
            collectors_seed,
            self.num_envs,
        )

    def warmup(self) -> None:
        """Trigger JIT compilation of select_action in the calling thread."""
        self.obs, _ = self.env.reset()
        self.select_action(self._rng, self.online_params[0], self.obs)

    def run_rollout(self) -> list[TimeStepDQN]:
        """Return one TimeStepDQN trajectory per environment."""
        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []
        reward_list: list[np.ndarray] = []
        terminated_list: list[np.ndarray] = []
        truncated_list: list[np.ndarray] = []

        for _ in range(self.steps_per_rollout):
            self._rng, action_jnp = self.select_action(
                self._rng,
                self._random.choice(self.online_params),
                self.obs,
            )
            actions = np.asarray(action_jnp, dtype=self.dtypes.action)  # (num_envs,)
            next_obs, rewards, terminated, truncated, _infos = self.env.step(actions)

            obs_list.append(self.obs)
            action_list.append(actions)
            reward_list.append(rewards)
            terminated_list.append(terminated)
            truncated_list.append(truncated)

            self.episode_reward_now += rewards
            done = np.logical_or(terminated, truncated)
            for i in np.where(done)[0]:
                self.completed_returns.append(float(self.episode_reward_now[i]))
                self.episode_reward_now[i] = 0.0

            # AutoresetMode.DISABLED: env.step() returns the terminal obs but
            # never resets sub-environments internally. Reset done envs here so
            # self.obs always holds a valid initial observation for the next step.
            if done.any():
                reset_obs, _ = self.env.reset(options={"reset_mask": done})
                next_obs = next_obs.copy()
                next_obs[done] = reset_obs[done]

            self.obs = next_obs

        # Stack to (T, N, ...) then swap to (N, T, ...) for per-env trajectories.
        obs_arr = np.stack(obs_list).astype(self.dtypes.obs).swapaxes(0, 1)
        act_arr = np.stack(action_list).astype(self.dtypes.action).swapaxes(0, 1)
        rew_arr = np.stack(reward_list).astype(self.dtypes.reward).swapaxes(0, 1)
        ter_arr = np.stack(terminated_list).astype(self.dtypes.terminated).swapaxes(0, 1)
        tru_arr = np.stack(truncated_list).astype(self.dtypes.truncated).swapaxes(0, 1)

        return [
            TimeStepDQN(
                obs=obs_arr[i],
                action=act_arr[i],
                reward=rew_arr[i],
                terminated=ter_arr[i],
                truncated=tru_arr[i],
            )
            for i in range(self.num_envs)
        ]

    def collecting(self):
        try:
            self._collecting()
        finally:
            self.close()

    def _collecting(self):
        if self.obs is None:
            self.obs, _ = self.env.reset()
        for rollouts_count in itertools.count(start=1, step=1):
            trajectories = self.run_rollout()

            for trajectory in trajectories:
                training_done = self.command_executor.call(self.controller, "is_done")
                if training_done:
                    self.logger.info("Done signal received; finishing.")
                    return

                while True:
                    adding_traj_done = self.command_executor.call(
                        self.trainer,
                        "add_traj_seq",
                        (self.collector_seed, trajectory),
                    )
                    if adding_traj_done:
                        break
                    time.sleep(0.01)

            parameters = self.command_executor.resolve(self.futures_parameters)
            if parameters is not None:
                self.online_params.append(parameters)
                self._params_received += 1
            self.futures_parameters = self.command_executor.submit(
                self.controller,
                "get_parameters",
            )

            if rollouts_count % self.report_period == 0:
                self.logger.info(
                    "Episode returns (%d): %s. Param updates: %d.",
                    len(self.completed_returns),
                    [f"{r:.1f}" for r in self.completed_returns],
                    self._params_received,
                )
                self.completed_returns.clear()
                self._params_received = 0

    def close(self) -> None:
        """Release local helper resources after collection stops."""
        if self._closed:
            return
        self._closed = True
        self.command_executor.shutdown()
        self.env.close()
        self.logger.info("CollectorDQNUniform closed (seed=%d).", self.collector_seed)


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
        self.logger = logging.getLogger(f"{__name__}.seed{collectors_seed}")
        self.logger.setLevel(log_level)
        self.report_period = report_period

        self.controller = controller
        self.trainer = trainer
        self.command_executor = CommandExecutor(max_workers=1)

        self.env = env(**args_env)

        self.collector_seed = collectors_seed
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
            "last": float("nan"),
        }
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError(
                "No GPU devices found. CollectorDreamerUniform requires at least one GPU."
            )
        self.gpu_device = gpu_devices[0]
        self.rollout_length = 1000
        self._closed = False

    def warmup(self) -> None:
        """Trigger JIT compilation of select_action in the calling thread."""
        action = {k: v[0] for k, v in self.action.items()}
        obs = self.env.step(action)
        obs = {k: np.stack([obs[k]]) for k in obs if not k.startswith("log/")}
        self.select_action(self.flax_state, obs)

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
                self.episode_reward["last"] = float(self.episode_reward["now"])
                self.episode_reward["now"] = 0

        trajectory = {k: np.concatenate([x[k] for x in trajectory], axis=0) for k in trajectory[0]}
        trajectory["consec"] = np.full(trajectory["is_first"].shape, 0, np.int32)
        return trajectory

    def collecting(self):
        try:
            self._collecting()
        finally:
            self.close()

    def _collecting(self):
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
                r = self.episode_reward["last"]
                self.logger.info(
                    "Last episode reward: %s.",
                    f"{r:.4f}" if not np.isnan(r) else "n/a",
                )

    def close(self) -> None:
        """Release local helper resources after collection stops."""
        if self._closed:
            return
        self._closed = True
        self.command_executor.shutdown()
        self.env.close()
