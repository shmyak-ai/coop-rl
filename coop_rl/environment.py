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
        self.observation_space = Box(shape=tuple(new_obs_space_shape), low=-np.inf, high=np.inf)

    def observation(self, obs):
        transposed_obs = np.moveaxis(obs, 0, -1)
        return transposed_obs


class HandlerEnv:
    def __init__(self, env_name, stack_size):
        self._env = HandlerEnv.make_env(env_name, stack_size)

    def reset(self, seed):
        observation, info = self._env.reset(seed=int(seed))
        # call [:] to get a stacked array from gym
        return observation[:], info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)
        return observation[:], reward, terminated, truncated, info

    @staticmethod
    def make_env(env_name, stack_size):
        env = gym.make(env_name)
        if stack_size > 1:
            env = FrameStackObservation(env, stack_size)
        return env

    @staticmethod
    def check_env(env_name, stack_size):
        env = HandlerEnv.make_env(env_name, stack_size)
        return (
            env.observation_space.shape,
            env.observation_space.dtype,
            env.action_space.n,
        )


class HandlerEnvAtari:
    def __init__(self, env_name, *args, stack_size=1, **kwargs):
        self._env = HandlerEnvAtari.make_env(env_name, stack_size, *args, **kwargs)

    def reset(self, *args, seed=None, **kwargs):
        if seed is not None:
            seed = int(seed)
        observation, info = self._env.reset(*args, seed=seed, **kwargs)
        return observation[:], info

    def step(self, action, *args, **kwargs):
        observation, reward, terminated, truncated, info = self._env.step(action, *args, **kwargs)
        return observation[:], reward, terminated, truncated, info

    @staticmethod
    def make_env(env_name, stack_size, *args, **kwargs):
        env = gym.make(env_name, *args, frameskip=1, repeat_action_probability=0, **kwargs)
        env = AtariPreprocessing(env, terminal_on_life_loss=False, grayscale_obs=True, scale_obs=True)
        if stack_size > 1:
            env = FrameStackObservation(env, stack_size)
            env = FirstDimToLast(env)
        return env

    @staticmethod
    def check_env(env_name, stack_size, *args, **kwargs):
        env = HandlerEnvAtari.make_env(env_name, stack_size, *args, **kwargs)
        return (
            env.observation_space.shape,
            env.observation_space.dtype,
            env.action_space.n,
        )

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def reward_range(self):
        return self._env.reward_range

    def close(self):
        self._env.close()