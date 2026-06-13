import os
from functools import partial

import elements
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


class FirstDimToLast(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_obs_space_shape = list(env.observation_space.shape[1:])
        new_obs_space_shape.append(env.observation_space.shape[0])
        orig = env.observation_space
        self.observation_space = Box(
            low=np.moveaxis(orig.low, 0, -1),
            high=np.moveaxis(orig.high, 0, -1),
            dtype=orig.dtype,
        )

    def observation(self, obs):
        transposed_obs = np.moveaxis(obs, 0, -1)
        return transposed_obs


class HandlerEnv:
    def __init__(self, env_name, stack_size):
        self._env = HandlerEnv.make_env(env_name, stack_size)

    def reset(self, *args, seed=None, **kwargs):
        observation, info = self._env.reset(*args, seed=seed, **kwargs)
        # call [:] to get a stacked array from gym
        return observation[:], info

    def step(self, action, *args, **kwargs):
        observation, reward, terminated, truncated, info = self._env.step(action)
        return observation[:], reward, terminated, truncated, info

    @staticmethod
    def make_env(env_name, stack_size):
        env = gym.make(env_name)
        if stack_size > 1:
            env = FrameStackObservation(env, stack_size)
        return env

    @staticmethod
    def check_env(env_name, stack_size, *args, **kwargs):
        env = HandlerEnv.make_env(env_name, stack_size)
        return (
            env.observation_space.shape,
            env.observation_space.dtype,
            env.action_space.n,
        )


def _make_atari_env(
    env_name,
    stack_size,
    kwargs,
    *,
    screen_size=84,
    grayscale=True,
    add_channel=False,
    repeat_action_probability=0.0,
):
    """Module-level factory for VectorEnv (must be picklable)."""
    import ale_py

    ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    env = gym.make(
        env_name, frameskip=1, repeat_action_probability=repeat_action_probability, **kwargs
    )
    env = AtariPreprocessing(
        env,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=grayscale,
        grayscale_newaxis=add_channel,
        scale_obs=False,
    )
    if stack_size > 1:
        env = FrameStackObservation(env, stack_size)
        env = FirstDimToLast(env)
    return env


class HandlerEnvAtari:
    """Atari env handler supporting 1 or more parallel environments.

    When ``num_envs=1`` a ``SyncVectorEnv`` is used (no subprocess overhead).
    When ``num_envs>1`` an ``AsyncVectorEnv`` is used (parallel subprocesses).
    ``reset`` and ``step`` always return batched arrays of shape
    ``(num_envs, *obs_shape)``.
    """

    def __init__(self, env_name, *, stack_size=1, num_envs=1, **kwargs):
        factory = partial(_make_atari_env, env_name, stack_size, kwargs)
        # DISABLED mode: env.step() never auto-resets sub-environments.
        # The collector resets done envs explicitly, preventing cross-episode
        # transitions from being stored in the replay buffer (which happens
        # with the default NextStep mode).
        autoreset_mode = gym.vector.AutoresetMode.DISABLED
        if num_envs > 1:
            self._env = gym.vector.AsyncVectorEnv(
                [factory] * num_envs, context="forkserver", autoreset_mode=autoreset_mode
            )
        else:
            self._env = gym.vector.SyncVectorEnv([factory], autoreset_mode=autoreset_mode)
        self.num_envs = num_envs

    def reset(self, *, seed=None, **kwargs):
        obs, info = self._env.reset(seed=seed, **kwargs)
        return obs, info  # (num_envs, *obs_shape)

    def step(self, actions):  # actions: (num_envs,)
        obs, rewards, terminated, truncated, infos = self._env.step(actions)
        return obs, rewards, terminated, truncated, infos

    @staticmethod
    def make_env(env_name, stack_size, **kwargs):
        return _make_atari_env(env_name, stack_size, kwargs)

    @staticmethod
    def check_env(env_name, stack_size, num_envs=1, **kwargs):
        env = _make_atari_env(env_name, stack_size, kwargs)
        shape = env.observation_space.shape
        dtype = env.observation_space.dtype
        n_actions = env.action_space.n
        env.close()
        return shape, dtype, n_actions

    def close(self):
        self._env.close()


class HandlerEnvDreamerAtari:
    """Gymnasium-based Atari env handler for DreamerV3.

    Single grayscale frame ``(screen_size, screen_size, 1)`` uint8, NEXT_STEP
    autoreset (the done step returns the terminal obs, the next step returns the
    fresh reset obs) so the collector can mark ``is_first`` from the previous
    step's done flag. ``check_env`` returns the dict-of-Spaces obs/act contract
    the Dreamer buffer and world model expect.
    """

    def __init__(self, *, env_name, num_envs=1, screen_size=96, sticky=True):
        rap = 0.25 if sticky else 0.0
        factory = partial(
            _make_atari_env,
            env_name,
            1,
            {},
            screen_size=screen_size,
            grayscale=True,
            add_channel=True,
            repeat_action_probability=rap,
        )
        autoreset_mode = gym.vector.AutoresetMode.NEXT_STEP
        if num_envs > 1:
            self._env = gym.vector.AsyncVectorEnv(
                [factory] * num_envs, context="forkserver", autoreset_mode=autoreset_mode
            )
        else:
            self._env = gym.vector.SyncVectorEnv([factory], autoreset_mode=autoreset_mode)
        self.num_envs = num_envs

    def reset(self, *, seed=None, **kwargs):
        obs, info = self._env.reset(seed=seed, **kwargs)
        return obs, info  # (num_envs, screen_size, screen_size, 1)

    def step(self, actions):  # actions: (num_envs,)
        return self._env.step(actions)  # obs, rewards, terminated, truncated, infos

    @staticmethod
    def make_env(env_name, screen_size=96, sticky=True):
        rap = 0.25 if sticky else 0.0
        return _make_atari_env(
            env_name,
            1,
            {},
            screen_size=screen_size,
            grayscale=True,
            add_channel=True,
            repeat_action_probability=rap,
        )

    def close(self):
        self._env.close()

    @staticmethod
    def check_env(*, env_name, num_envs=1, screen_size=96, sticky=True):
        env = HandlerEnvDreamerAtari.make_env(env_name, screen_size=screen_size, sticky=sticky)
        shape = env.observation_space.shape
        n_actions = env.action_space.n
        env.close()
        obs_space = {
            "image": elements.Space(np.uint8, shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        act_space = {"action": elements.Space(np.int32, (), 0, int(n_actions))}
        return obs_space, None, act_space


def _make_unity_env(file_name, base_port, worker_id, no_graphics, time_scale, seed):
    """Launch a UnityEnvironment, retrying nearby worker ids on collision.

    Imports ``mlagents_envs`` lazily so this module stays importable without the
    optional dependency (mirrors the ``import ale_py`` pattern in
    ``_make_atari_env``).
    """
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.exception import UnityWorkerInUseException
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )

    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(time_scale=time_scale)
    # Multiple collectors share one args_env, so a fixed worker_id collides.
    # Spread the starting offset by pid and probe upward for a free port.
    start = os.getpid() % 1000 if worker_id is None else worker_id
    for offset in range(128):
        try:
            return UnityEnvironment(
                file_name=file_name,
                worker_id=start + offset,
                base_port=base_port,
                no_graphics=no_graphics,
                seed=seed,
                side_channels=[engine],
            )
        except UnityWorkerInUseException:
            continue
    raise RuntimeError(
        f"No free Unity worker port found near base_port={base_port}, start={start}."
    )


def _single_behavior_name(env) -> str:
    names = list(env.behavior_specs.keys())
    if len(names) != 1:
        raise ValueError(f"HandlerUnityEnv supports exactly one behavior; found {names}.")
    return names[0]


def _find_visual_obs_index(observation_specs) -> int:
    for i, spec in enumerate(observation_specs):
        if len(spec.shape) == 3:
            return i
    raise ValueError(
        "HandlerUnityEnv requires a visual (3-D) observation; the behavior exposes "
        f"only shapes {[tuple(s.shape) for s in observation_specs]}."
    )


def _to_uint8_image(arr):
    # Unity visual observations are float32 in [0, 1]; Dreamer expects uint8 [0, 255].
    return (255.0 * arr).astype(np.uint8)


class HandlerUnityEnv:
    """Native multi-agent ML-Agents handler for the Dreamer pipeline.

    One ``UnityEnvironment`` binary; ``num_envs`` equals the number of agents in
    the scene. The single visual observation is returned as a uint8 ``image``.
    Episode resets use Unity's internal per-agent reset, surfaced here as
    ``NEXT_STEP`` autoreset (the terminal step returns the terminal obs with
    ``done=True``; the next step returns the new episode's first obs), matching
    what ``CollectorDreamerUniform`` expects.

    Agents are mapped to fixed batch slots; an agent keeps its slot for the whole
    episode (agent ids change on reset, so slots — not ids — are the stable key).
    Assumes a single behavior with decision period 1 (every live agent acts every
    step) and purely discrete (single branch) or purely continuous actions; it
    fails loudly otherwise. ``no_graphics`` defaults to ``False`` because visual
    observations require rendering.
    """

    def __init__(
        self,
        *,
        file_name,
        num_envs,
        base_port=5005,
        worker_id=None,
        no_graphics=False,
        time_scale=20.0,
        seed=0,
    ):
        self.num_envs = num_envs
        self._env = _make_unity_env(file_name, base_port, worker_id, no_graphics, time_scale, seed)
        self._env.reset()
        self._behavior_name = _single_behavior_name(self._env)
        spec = self._env.behavior_specs[self._behavior_name]
        self._vis_index = _find_visual_obs_index(spec.observation_specs)
        self._image_shape = tuple(spec.observation_specs[self._vis_index].shape)

        action_spec = spec.action_spec
        if action_spec.is_discrete():
            if action_spec.discrete_size != 1:
                raise ValueError(
                    "HandlerUnityEnv supports single-branch discrete actions only; "
                    f"got branches {action_spec.discrete_branches}."
                )
            self._discrete = True
            self._cont_size = 0
        elif action_spec.is_continuous():
            self._discrete = False
            self._cont_size = int(action_spec.continuous_size)
        else:
            raise ValueError(
                "HandlerUnityEnv supports purely discrete or purely continuous actions; "
                f"got branches={action_spec.discrete_branches}, "
                f"continuous={action_spec.continuous_size}."
            )

        # agent_id -> batch slot for currently living agents; slots freed by a
        # termination wait in _vacant until a new agent claims them.
        self._id2slot: dict[int, int] = {}
        self._vacant: set[int] = set()
        self._dec_ids: list[int] = []  # agent ids in the pending decision batch

    def reset(self, *, seed=None, **kwargs):
        self._env.reset()
        decision, terminal = self._env.get_steps(self._behavior_name)
        if len(terminal) != 0:
            raise RuntimeError("Unexpected terminal steps immediately after reset.")
        if len(decision) != self.num_envs:
            raise RuntimeError(
                f"Unity scene has {len(decision)} agents requesting decisions but "
                f"num_envs={self.num_envs}; they must match (decision period must be 1)."
            )
        self._id2slot = {int(aid): i for i, aid in enumerate(decision.agent_id)}
        self._vacant = set()
        self._dec_ids = [int(aid) for aid in decision.agent_id]
        image = np.empty((self.num_envs, *self._image_shape), np.uint8)
        vis = decision.obs[self._vis_index]
        for k, aid in enumerate(self._dec_ids):
            image[self._id2slot[aid]] = _to_uint8_image(vis[k])
        return image.copy(), {}

    def step(self, actions):
        from mlagents_envs.base_env import ActionTuple

        actions = np.asarray(actions)
        n_dec = len(self._dec_ids)
        action_tuple = ActionTuple()
        if self._discrete:
            arr = np.zeros((n_dec, 1), np.int32)
            for k, aid in enumerate(self._dec_ids):
                arr[k, 0] = int(actions[self._id2slot[aid]])
            action_tuple.add_discrete(arr)
        else:
            arr = np.zeros((n_dec, self._cont_size), np.float32)
            for k, aid in enumerate(self._dec_ids):
                arr[k] = actions[self._id2slot[aid]]
            action_tuple.add_continuous(arr)
        self._env.set_actions(self._behavior_name, action_tuple)
        self._env.step()
        decision, terminal = self._env.get_steps(self._behavior_name)

        image = np.empty((self.num_envs, *self._image_shape), np.uint8)
        reward = np.zeros(self.num_envs, np.float32)
        terminated = np.zeros(self.num_envs, bool)
        truncated = np.zeros(self.num_envs, bool)
        filled = np.zeros(self.num_envs, bool)

        # Terminations first: record the terminal obs and free the slot.
        freed_this_step: list[int] = []
        term_vis = terminal.obs[self._vis_index]
        for k, raw in enumerate(terminal.agent_id):
            slot = self._id2slot.pop(int(raw))
            image[slot] = _to_uint8_image(term_vis[k])
            reward[slot] = terminal.reward[k]
            interrupted = bool(terminal.interrupted[k])
            terminated[slot] = not interrupted
            truncated[slot] = interrupted
            filled[slot] = True
            freed_this_step.append(slot)

        # Decisions: continuing agents keep their slot; a new (reset) agent claims
        # a slot vacated on a *previous* step (NEXT_STEP). No prior vacancy means
        # the env surfaced the reset decision in the same tick as the termination
        # (decision timing this handler does not support).
        dec_vis = decision.obs[self._vis_index]
        new_dec_ids: list[int] = []
        for k, raw in enumerate(decision.agent_id):
            aid = int(raw)
            if aid in self._id2slot:
                slot = self._id2slot[aid]
            elif self._vacant:
                slot = self._vacant.pop()
                self._id2slot[aid] = slot
            else:
                raise RuntimeError(
                    "A reset agent appeared without a free slot; HandlerUnityEnv "
                    "requires terminal and the following decision to be one step "
                    "apart (decision period 1)."
                )
            image[slot] = _to_uint8_image(dec_vis[k])
            reward[slot] = decision.reward[k]
            filled[slot] = True
            new_dec_ids.append(aid)

        if not filled.all():
            raise RuntimeError(
                "Some envs reported neither a decision nor a terminal step this tick; "
                "HandlerUnityEnv requires decision period 1 (every agent acts every step)."
            )
        self._vacant.update(freed_this_step)
        self._dec_ids = new_dec_ids
        return image, reward, terminated, truncated, {}

    def close(self):
        self._env.close()

    @staticmethod
    def check_env(
        *,
        file_name,
        num_envs,
        base_port=5005,
        worker_id=None,
        no_graphics=False,
        time_scale=20.0,
        seed=0,
    ):
        env = _make_unity_env(file_name, base_port, worker_id, no_graphics, time_scale, seed)
        try:
            env.reset()
            behavior_name = _single_behavior_name(env)
            spec = env.behavior_specs[behavior_name]
            vis_index = _find_visual_obs_index(spec.observation_specs)
            shape = tuple(spec.observation_specs[vis_index].shape)
            decision, _ = env.get_steps(behavior_name)
            n_agents = len(decision)
            action_spec = spec.action_spec
            if action_spec.is_discrete():
                if action_spec.discrete_size != 1:
                    raise ValueError(
                        "HandlerUnityEnv supports single-branch discrete actions only; "
                        f"got branches {action_spec.discrete_branches}."
                    )
                act_space = {
                    "action": elements.Space(np.int32, (), 0, int(action_spec.discrete_branches[0]))
                }
            elif action_spec.is_continuous():
                act_space = {
                    "action": elements.Space(
                        np.float32, (int(action_spec.continuous_size),), -1.0, 1.0
                    )
                }
            else:
                raise ValueError(
                    "HandlerUnityEnv supports purely discrete or purely continuous actions; "
                    f"got branches={action_spec.discrete_branches}, "
                    f"continuous={action_spec.continuous_size}."
                )
        finally:
            env.close()
        if n_agents != num_envs:
            raise ValueError(
                f"Unity scene has {n_agents} agents but num_envs={num_envs}; they must match."
            )
        obs_space = {
            "image": elements.Space(np.uint8, shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        return obs_space, None, act_space
