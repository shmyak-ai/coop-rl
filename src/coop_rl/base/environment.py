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
