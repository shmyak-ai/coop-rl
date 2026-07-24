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
import os
import random
import time
from collections import deque

import elements
import jax
import numpy as np

from coop_rl.base.base_types import TimeStepDQN, TimeStepDQNRecurrent
from coop_rl.networks.base import ScannedRNN
from coop_rl.workers.auxiliary import CommandExecutor, _TBWriter


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
        workdir: str | None = None,
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
        self._env_steps = 0
        self._writer: _TBWriter | None = (
            _TBWriter(os.path.join(workdir, "tb")) if workdir is not None else None
        )
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

    def set_online_params(self, params) -> None:
        """Replace, not append: sequential training acts with the latest params only."""
        self.online_params.clear()
        self.online_params.append(params)

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
        obs_arr = np.stack(obs_list)
        del obs_list
        obs_arr = obs_arr.astype(self.dtypes.obs).swapaxes(0, 1)
        act_arr = np.stack(action_list).astype(self.dtypes.action).swapaxes(0, 1)
        rew_arr = np.stack(reward_list).astype(self.dtypes.reward).swapaxes(0, 1)
        ter_arr = np.stack(terminated_list).astype(self.dtypes.terminated).swapaxes(0, 1)
        tru_arr = np.stack(truncated_list).astype(self.dtypes.truncated).swapaxes(0, 1)

        return TimeStepDQN(
            obs=obs_arr,
            action=act_arr,
            reward=rew_arr,
            terminated=ter_arr,
            truncated=tru_arr,
        )

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
            self._env_steps += self.steps_per_rollout * self.num_envs

            training_done = self.command_executor.call(self.controller, "is_done")
            if training_done:
                self.logger.info("Done signal received; finishing.")
                return

            while True:
                adding_traj_done = self.command_executor.call(
                    self.trainer,
                    "add_traj_seq",
                    (self.collector_seed, trajectories),
                )
                if adding_traj_done:
                    break
                time.sleep(0.01)
            del trajectories

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
                if self._writer is not None and self.completed_returns:
                    self._writer.write_scalars(
                        self._env_steps,
                        {"collector/mean_return": float(np.mean(self.completed_returns))},
                    )
                    self._writer.flush()
                self.completed_returns.clear()
                self._params_received = 0

    def close(self) -> None:
        """Release local helper resources after collection stops."""
        if self._closed:
            return
        self._closed = True
        self.command_executor.shutdown()
        self.env.close()
        if self._writer is not None:
            self._writer.close()
        self.logger.info("CollectorDQNUniform closed (seed=%d).", self.collector_seed)


class CollectorDQNRecurrentUniform:
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
        hidden_state_dim,
        cell_type,
        workdir: str | None = None,
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

        self.hidden_state_dim = hidden_state_dim
        self.cell_type = cell_type
        carry = ScannedRNN(hidden_state_dim, cell_type).initialize_carry(1)
        if not isinstance(carry, jax.Array):
            raise ValueError(
                f"cell_type '{cell_type}' has a non-array carry; TimeStepDQNRecurrent "
                "only supports single-array carries (e.g. GRU)."
            )

        self.collector_seed = collectors_seed
        self._rng = jax.random.PRNGKey(collectors_seed)
        self._rng, rng = jax.random.split(self._rng)
        args_state_recover.rng = rng
        flax_state = state_recover(**args_state_recover)

        # latest params only: stale params would clash with the stored hidden states
        self.online_params = flax_state.params

        self.futures_parameters = self.command_executor.submit(self.controller, "get_parameters")

        args_get_select_action_fn.apply_fn = flax_state.apply_fn
        self.select_action = get_select_action_fn(**args_get_select_action_fn)
        self.obs = None
        self.hidden_state = None
        self.reset_mask = None
        self.prev_action = None
        self.prev_reward = None
        self.episode_reward_now = np.zeros(self.num_envs)
        self.completed_returns: deque[float] = deque(maxlen=100)
        self._params_received = 0
        self._env_steps = 0
        self._writer: _TBWriter | None = (
            _TBWriter(os.path.join(workdir, "tb")) if workdir is not None else None
        )
        self._closed = False
        self.logger.info(
            "CollectorDQNRecurrentUniform initialized (seed=%d, num_envs=%d).",
            collectors_seed,
            self.num_envs,
        )

    def _reset_obs(self) -> None:
        self.obs, _ = self.env.reset()
        self.hidden_state = ScannedRNN(self.hidden_state_dim, self.cell_type).initialize_carry(
            self.num_envs
        )
        self.reset_mask = np.ones(self.num_envs, dtype=bool)
        self.prev_action = np.zeros(self.num_envs, dtype=self.dtypes.prev_action)
        self.prev_reward = np.zeros(self.num_envs, dtype=self.dtypes.prev_reward)

    def warmup(self) -> None:
        """Trigger JIT compilation of select_action in the calling thread."""
        self._reset_obs()
        self.select_action(
            self._rng,
            self.online_params,
            self.hidden_state,
            self.obs,
            self.reset_mask,
            self.prev_action,
            self.prev_reward,
        )

    def set_online_params(self, params) -> None:
        self.online_params = params

    def run_rollout(self) -> TimeStepDQNRecurrent:
        """Return one TimeStepDQNRecurrent trajectory per environment."""
        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []
        reward_list: list[np.ndarray] = []
        terminated_list: list[np.ndarray] = []
        truncated_list: list[np.ndarray] = []
        hidden_state_list: list[np.ndarray] = []
        reset_mask_list: list[np.ndarray] = []
        prev_action_list: list[np.ndarray] = []
        prev_reward_list: list[np.ndarray] = []

        params = self.online_params

        for _ in range(self.steps_per_rollout):
            hidden_state_before = self.hidden_state
            reset_mask_before = self.reset_mask
            prev_action_before = self.prev_action
            prev_reward_before = self.prev_reward

            self._rng, self.hidden_state, action_jnp = self.select_action(
                self._rng,
                params,
                self.hidden_state,
                self.obs,
                self.reset_mask,
                self.prev_action,
                self.prev_reward,
            )
            actions = np.asarray(action_jnp, dtype=self.dtypes.action)  # (num_envs,)
            next_obs, rewards, terminated, truncated, _infos = self.env.step(actions)

            obs_list.append(self.obs)
            action_list.append(actions)
            reward_list.append(rewards)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            hidden_state_list.append(
                np.asarray(hidden_state_before, dtype=self.dtypes.hidden_state)
            )
            reset_mask_list.append(reset_mask_before.astype(self.dtypes.reset_hidden_state))
            prev_action_list.append(np.asarray(prev_action_before, dtype=self.dtypes.prev_action))
            prev_reward_list.append(np.asarray(prev_reward_before, dtype=self.dtypes.prev_reward))

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
            self.reset_mask = done
            # The next episode's first step has no predecessor: zero prev inputs on done.
            self.prev_action = np.where(done, 0, actions).astype(self.dtypes.prev_action)
            self.prev_reward = np.where(done, 0, rewards).astype(self.dtypes.prev_reward)

        # Stack to (T, N, ...) then swap to (N, T, ...) for per-env trajectories.
        obs_arr = np.stack(obs_list)
        del obs_list
        obs_arr = obs_arr.astype(self.dtypes.obs).swapaxes(0, 1)
        act_arr = np.stack(action_list).astype(self.dtypes.action).swapaxes(0, 1)
        rew_arr = np.stack(reward_list).astype(self.dtypes.reward).swapaxes(0, 1)
        ter_arr = np.stack(terminated_list).astype(self.dtypes.terminated).swapaxes(0, 1)
        tru_arr = np.stack(truncated_list).astype(self.dtypes.truncated).swapaxes(0, 1)
        hid_arr = np.stack(hidden_state_list).astype(self.dtypes.hidden_state).swapaxes(0, 1)
        rst_arr = np.stack(reset_mask_list).astype(self.dtypes.reset_hidden_state).swapaxes(0, 1)
        pact_arr = np.stack(prev_action_list).astype(self.dtypes.prev_action).swapaxes(0, 1)
        prew_arr = np.stack(prev_reward_list).astype(self.dtypes.prev_reward).swapaxes(0, 1)

        return TimeStepDQNRecurrent(
            obs=obs_arr,
            action=act_arr,
            reward=rew_arr,
            terminated=ter_arr,
            truncated=tru_arr,
            hidden_state=hid_arr,
            reset_hidden_state=rst_arr,
            prev_action=pact_arr,
            prev_reward=prew_arr,
        )

    def collecting(self):
        try:
            self._collecting()
        finally:
            self.close()

    def _collecting(self):
        if self.obs is None:
            self._reset_obs()
        for rollouts_count in itertools.count(start=1, step=1):
            trajectories = self.run_rollout()
            self._env_steps += self.steps_per_rollout * self.num_envs

            while True:
                training_done = self.command_executor.call(self.controller, "is_done")
                if training_done:
                    self.logger.info("Done signal received; finishing.")
                    return

                adding_traj_done = self.command_executor.call(
                    self.trainer,
                    "add_traj_seq",
                    (self.collector_seed, trajectories),
                )
                if adding_traj_done:
                    break
                time.sleep(0.01)
            del trajectories

            parameters = self.command_executor.resolve(self.futures_parameters)
            if parameters is not None:
                self.online_params = jax.device_put(parameters)
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
                if self._writer is not None and self.completed_returns:
                    self._writer.write_scalars(
                        self._env_steps,
                        {"collector/mean_return": float(np.mean(self.completed_returns))},
                    )
                    self._writer.flush()
                self.completed_returns.clear()
                self._params_received = 0

    def close(self) -> None:
        """Release local helper resources after collection stops."""
        if self._closed:
            return
        self._closed = True
        self.command_executor.shutdown()
        self.env.close()
        if self._writer is not None:
            self._writer.close()
        self.logger.info("CollectorDQNRecurrentUniform closed (seed=%d).", self.collector_seed)


class CollectorDQNWindowedUniform:
    """Non-recurrent collector for a causal-transformer network with a fixed
    context_length: unlike CollectorDQNRecurrentUniform's O(1) hidden-state
    carry, the transformer needs the actual last context_length transitions to
    act, so select_action is handed a sliding window (obs/action/reward/
    terminated/truncated), not a single step. Only single-timestep TimeStepDQN
    transitions are stored to the buffer (via run_rollout, like
    CollectorDQNUniform) -- the window is a collector-local view for action
    selection only, never persisted, so there is no per-transition storage
    blowup from the window length.
    """

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
        context_length,
        workdir: str | None = None,
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
        self.context_length = context_length

        self.collector_seed = collectors_seed
        self._random = random.Random(collectors_seed)
        self._rng = jax.random.PRNGKey(collectors_seed)
        self._rng, rng = jax.random.split(self._rng)
        args_state_recover.rng = rng
        flax_state = state_recover(**args_state_recover)

        # online params are to prevent dqn algs from freezing; no stored
        # hidden-state staleness concern here (the window holds raw env data,
        # not param-derived activations), so this mirrors CollectorDQNUniform
        # rather than CollectorDQNRecurrentUniform's single-latest-params rule.
        self.online_params = deque(maxlen=10)
        self.online_params.append(flax_state.params)

        self.futures_parameters = self.command_executor.submit(self.controller, "get_parameters")

        args_get_select_action_fn.apply_fn = flax_state.apply_fn
        self.select_action = get_select_action_fn(**args_get_select_action_fn)
        self.obs = None
        self._win_obs = None
        self._win_action = None
        self._win_reward = None
        self._win_terminated = None
        self._win_truncated = None
        self.episode_reward_now = np.zeros(self.num_envs)
        self.completed_returns: deque[float] = deque(maxlen=100)
        self._params_received = 0
        self._env_steps = 0
        self._writer: _TBWriter | None = (
            _TBWriter(os.path.join(workdir, "tb")) if workdir is not None else None
        )
        self._closed = False
        self.logger.info(
            "CollectorDQNWindowedUniform initialized (seed=%d, num_envs=%d, context_length=%d).",
            collectors_seed,
            self.num_envs,
            self.context_length,
        )

    def _reset_obs(self) -> None:
        self.obs, _ = self.env.reset()
        win_shape = (self.num_envs, self.context_length, *self.obs.shape[1:])
        self._win_obs = np.zeros(win_shape, dtype=self.dtypes.obs)
        self._win_action = np.zeros((self.num_envs, self.context_length), dtype=self.dtypes.action)
        self._win_reward = np.zeros((self.num_envs, self.context_length), dtype=self.dtypes.reward)
        self._win_terminated = np.zeros(
            (self.num_envs, self.context_length), dtype=self.dtypes.terminated
        )
        # Seed every slot but the newest as "truncated" so build_causal_boundary_mask
        # (an exclusive cumsum of terminated|truncated) treats each not-yet-real
        # placeholder as its own isolated singleton episode -- unreachable from the
        # newest (real) slot and from each other -- with no separate padding/
        # validity concept needed anywhere else in the stack.
        self._win_truncated = np.ones(
            (self.num_envs, self.context_length), dtype=self.dtypes.truncated
        )
        self._win_obs[:, -1] = self.obs
        self._win_truncated[:, -1] = 0

    def _push_window(self, next_obs: np.ndarray) -> None:
        for arr in (
            self._win_obs,
            self._win_action,
            self._win_reward,
            self._win_terminated,
            self._win_truncated,
        ):
            arr[:, :-1] = arr[:, 1:]
        self._win_obs[:, -1] = next_obs
        self._win_action[:, -1] = 0
        self._win_reward[:, -1] = 0
        self._win_terminated[:, -1] = 0
        self._win_truncated[:, -1] = 0

    def warmup(self) -> None:
        """Trigger JIT compilation of select_action in the calling thread."""
        self._reset_obs()
        self.select_action(
            self._rng,
            self.online_params[0],
            self._win_obs,
            self._win_action,
            self._win_reward,
            self._win_terminated,
            self._win_truncated,
        )

    def set_online_params(self, params) -> None:
        """Replace, not append: sequential training acts with the latest params only."""
        self.online_params.clear()
        self.online_params.append(params)

    def run_rollout(self) -> TimeStepDQN:
        """Return one TimeStepDQN trajectory per environment, selecting each
        action from the current context_length window (updated internally
        every step) rather than from a single-step observation."""
        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []
        reward_list: list[np.ndarray] = []
        terminated_list: list[np.ndarray] = []
        truncated_list: list[np.ndarray] = []

        for _ in range(self.steps_per_rollout):
            self._rng, action_jnp = self.select_action(
                self._rng,
                self._random.choice(self.online_params),
                self._win_obs,
                self._win_action,
                self._win_reward,
                self._win_terminated,
                self._win_truncated,
            )
            actions = np.asarray(action_jnp, dtype=self.dtypes.action)  # (num_envs,)
            next_obs, rewards, terminated, truncated, _infos = self.env.step(actions)

            # Fill in the transition just taken at the window's newest slot
            # (it held only a placeholder obs until now, since the action from
            # it wasn't chosen yet when the window was last pushed).
            self._win_action[:, -1] = actions
            self._win_reward[:, -1] = rewards.astype(self.dtypes.reward)
            self._win_terminated[:, -1] = terminated.astype(self.dtypes.terminated)
            self._win_truncated[:, -1] = truncated.astype(self.dtypes.truncated)

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

            self._push_window(next_obs)
            self.obs = next_obs

        # Stack to (T, N, ...) then swap to (N, T, ...) for per-env trajectories.
        obs_arr = np.stack(obs_list)
        del obs_list
        obs_arr = obs_arr.astype(self.dtypes.obs).swapaxes(0, 1)
        act_arr = np.stack(action_list).astype(self.dtypes.action).swapaxes(0, 1)
        rew_arr = np.stack(reward_list).astype(self.dtypes.reward).swapaxes(0, 1)
        ter_arr = np.stack(terminated_list).astype(self.dtypes.terminated).swapaxes(0, 1)
        tru_arr = np.stack(truncated_list).astype(self.dtypes.truncated).swapaxes(0, 1)

        return TimeStepDQN(
            obs=obs_arr,
            action=act_arr,
            reward=rew_arr,
            terminated=ter_arr,
            truncated=tru_arr,
        )

    def collecting(self):
        try:
            self._collecting()
        finally:
            self.close()

    def _collecting(self):
        if self.obs is None:
            self._reset_obs()
        for rollouts_count in itertools.count(start=1, step=1):
            trajectories = self.run_rollout()
            self._env_steps += self.steps_per_rollout * self.num_envs

            training_done = self.command_executor.call(self.controller, "is_done")
            if training_done:
                self.logger.info("Done signal received; finishing.")
                return

            while True:
                adding_traj_done = self.command_executor.call(
                    self.trainer,
                    "add_traj_seq",
                    (self.collector_seed, trajectories),
                )
                if adding_traj_done:
                    break
                time.sleep(0.01)
            del trajectories

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
                if self._writer is not None and self.completed_returns:
                    self._writer.write_scalars(
                        self._env_steps,
                        {"collector/mean_return": float(np.mean(self.completed_returns))},
                    )
                    self._writer.flush()
                self.completed_returns.clear()
                self._params_received = 0

    def close(self) -> None:
        """Release local helper resources after collection stops."""
        if self._closed:
            return
        self._closed = True
        self.command_executor.shutdown()
        self.env.close()
        if self._writer is not None:
            self._writer.close()
        self.logger.info("CollectorDQNWindowedUniform closed (seed=%d).", self.collector_seed)


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
        workdir: str | None = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.seed{collectors_seed}")
        self.logger.setLevel(log_level)
        self.report_period = report_period

        self.controller = controller
        self.trainer = trainer
        self.command_executor = CommandExecutor(max_workers=1)

        self.env = env(**args_env)
        self.num_envs = self.env.num_envs

        self.collector_seed = collectors_seed
        args_state_recover["rng"] = jax.random.PRNGKey(collectors_seed)
        self.flax_state = state_recover(**args_state_recover)

        self.futures_parameters = self.command_executor.submit(self.controller, "get_parameters")
        self.select_action = get_select_action_fn(self.flax_state)
        self.obs = None
        self.prev_done = np.zeros(self.num_envs, bool)
        self.episode_reward_now = np.zeros(self.num_envs)
        self.completed_returns: deque[float] = deque(maxlen=100)
        self._params_received = 0
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError(
                "No GPU devices found. CollectorDreamerUniform requires at least one GPU."
            )
        self.gpu_device = gpu_devices[0]
        self.rollout_length = 100
        self._env_steps = 0
        self._writer: _TBWriter | None = (
            _TBWriter(os.path.join(workdir, "tb")) if workdir is not None else None
        )
        self._closed = False

    def _reset_obs(self) -> None:
        img, _ = self.env.reset(seed=self.collector_seed)
        self.obs = {
            "image": img,
            "is_first": np.ones(self.num_envs, bool),
            "is_last": np.zeros(self.num_envs, bool),
            "is_terminal": np.zeros(self.num_envs, bool),
            "reward": np.zeros(self.num_envs, np.float32),
        }
        self.prev_done = np.zeros(self.num_envs, bool)

    def warmup(self) -> None:
        """Trigger JIT compilation of select_action in the calling thread."""
        self._reset_obs()
        self.select_action(self.flax_state, self.obs)

    def run_rollout(self):
        trajectory = []
        uuids = [elements.UUID() for _ in range(self.num_envs)]
        for index in range(self.rollout_length):
            self.flax_state, action, outs = self.select_action(self.flax_state, self.obs)
            act = np.asarray(action["action"], dtype=np.int32)

            step_id = np.stack(
                [np.frombuffer(bytes(u) + index.to_bytes(4, "big"), np.uint8) for u in uuids]
            )
            trajectory.append(
                {
                    "image": self.obs["image"],
                    "is_first": self.obs["is_first"],
                    "is_last": self.obs["is_last"],
                    "is_terminal": self.obs["is_terminal"],
                    "reward": self.obs["reward"],
                    "stepid": step_id,
                    "dyn/deter": outs["dyn/deter"],
                    "dyn/stoch": outs["dyn/stoch"],
                    "action": act,
                }
            )

            # NEXT_STEP autoreset: the step that finishes an episode returns the
            # terminal obs (done=True); the following step returns the fresh reset
            # obs (action ignored). So the new obs is a first-of-episode frame
            # exactly for the envs that were done on the previous step.
            next_img, reward, terminated, truncated, _ = self.env.step(act)
            done = np.logical_or(terminated, truncated)
            is_first = self.prev_done
            reward = np.where(is_first, 0.0, reward).astype(np.float32)
            self.obs = {
                "image": next_img,
                "is_first": is_first,
                "is_last": done,
                "is_terminal": done,
                "reward": reward,
            }
            self.prev_done = done

            self.episode_reward_now += reward
            for i in np.where(done)[0]:
                self.completed_returns.append(float(self.episode_reward_now[i]))
                self.episode_reward_now[i] = 0.0

        trajectory = {k: np.stack([x[k] for x in trajectory], axis=1) for k in trajectory[0]}
        trajectory["consec"] = np.full(trajectory["is_first"].shape, 0, np.int32)
        return trajectory

    def collecting(self):
        try:
            self._collecting()
        finally:
            self.close()

    def _collecting(self):
        if self.obs is None:
            self._reset_obs()
        for rollouts_count in itertools.count(start=1, step=1):
            trajectory = self.run_rollout()
            self._env_steps += self.rollout_length * self.num_envs

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
                if self._writer is not None and self.completed_returns:
                    self._writer.write_scalars(
                        self._env_steps,
                        {"collector/mean_return": float(np.mean(self.completed_returns))},
                    )
                    self._writer.flush()
                self.completed_returns.clear()
                self._params_received = 0

    def close(self) -> None:
        """Release local helper resources after collection stops."""
        if self._closed:
            return
        self._closed = True
        self.command_executor.shutdown()
        self.env.close()
        if self._writer is not None:
            self._writer.close()
