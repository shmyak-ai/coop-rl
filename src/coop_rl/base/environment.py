from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import coop_rl.dreamer.wrappers as wrappers
from coop_rl.dreamer.envs.atari import Atari


def wrap_env(env):
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.NormalizeAction(env, name)
    env = wrappers.UnifyDtypes(env)
    env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env


class FirstDimToLast(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_obs_space_shape = list(env.observation_space.shape[1:])
        new_obs_space_shape.append(env.observation_space.shape[0])
        self.observation_space = Box(shape=tuple(new_obs_space_shape), low=-np.inf, high=np.inf)

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


def _make_atari_env(env_name, stack_size, kwargs):
    """Module-level factory for VectorEnv (must be picklable)."""
    import ale_py

    ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    env = gym.make(env_name, frameskip=1, repeat_action_probability=0, **kwargs)
    env = AtariPreprocessing(env, terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False)
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
        if num_envs > 1:
            self._env = gym.vector.AsyncVectorEnv([factory] * num_envs, context="forkserver")
        else:
            self._env = gym.vector.SyncVectorEnv([factory])
        self.num_envs = num_envs

    def reset(self, *, seed=None, **kwargs):
        obs, info = self._env.reset(seed=seed)
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
    def __init__(self, *, dreamer_config):
        self._env = HandlerEnvDreamerAtari.make_env(dreamer_config)

    def reset(self, *args, **kwargs):
        return None

    def step(self, action, *args, **kwargs):
        return self._env.step(action)

    @staticmethod
    def make_env(config):
        suite, task = config.task.split("_", 1)
        kwargs = config.env.get(suite, {})
        kwargs.update({})
        env = Atari(task, **kwargs)
        return wrap_env(env)

    @staticmethod
    def check_env(*, dreamer_config):
        env = HandlerEnvDreamerAtari.make_env(dreamer_config)
        obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith("log/")}
        act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
        env.close()

        return (
            obs_space,
            None,
            act_space,
        )
