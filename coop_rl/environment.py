from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers.frame_stack import LazyFrames


class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `num_stack` many identical frames.

    Note2: Derived from the gymnasium FrameStack, but to follow Dopamine this implementation
        puts zeros in the beginning of an episode.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStack
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.utils.RecordConstructorArgs.__init__(self, num_stack=num_stack, lz4_compress=lz4_compress)
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)

        zeros = np.zeros_like(obs)
        [self.frames.append(zeros) for _ in range(self.num_stack - 1)]
        self.frames.append(obs)

        return self.observation(None), info


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
            env = FrameStack(env, stack_size)
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
            env = FrameStack(env, stack_size)
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