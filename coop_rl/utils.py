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

import functools
import math
import time
from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import reverb
import tensorflow as tf
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers.frame_stack import LazyFrames

from coop_rl import networks
from coop_rl.agents import dqn


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


class HandlerEnvAtariDopamine:
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


class HandlerDopamineReplay:
    def __init__(self, stack_size):
        self._replay = []
        self._stack_size = stack_size

    def reset(self):
        self._replay = []

    def _store_transition(self, last_observation, action, reward, terminated, *args, priority=None, truncated=False):
        """Stores a transition when in training mode.

        Stores the following tuple in the replay buffer (last_observation, action,
        reward, is_terminal, priority). Follows the dopamine replay buffer naming.

        Args:
          last_observation: Last observation, type determined via observation_type
            parameter in the replay_memory constructor.
          action: An integer, the action taken after the last observation.
          reward: A float, the reward received for the action after the last observation.
          terminated: Boolean indicating if the current state is a terminal state.
          Or terminal in dopamine. Similar to gymnasium terminated.
          *args: Any, other items to be added to the replay buffer.
          priority: Float. Priority of sampling the transition. If None, the default
            priority will be used. If replay scheme is uniform, the default priority
            is 1. If the replay scheme is prioritized, the default priority is the
            maximum ever seen [Schaul et al., 2015].
          truncated: bool, whether this transition is the last for the episode.
            This can be different than terminal when ending the episode because of a
            timeout. Episode_end in dopamine. Similar to gymnasium truncated.
        """
        if self._stack_size > 1:
            last_observation = last_observation[..., -1]

        self._replay.append(
            (
                last_observation,
                action,
                reward,
                terminated,
                *args,
                {
                    "priority": priority,
                    "truncated": truncated,
                },
            )
        )

    @property
    def replay(self):
        return self._replay

    def close(self):
        pass


class HandlerReverbReplay:
    def __init__(self, timesteps: int, table_name: str, ip: str = "localhost", buffer_server_port: int = 8023):
        """
        Args:
            timesteps
            table_name
            ip and buffer_server_port: this is server adress.
        """
        self.timesteps = timesteps
        self.table_name = table_name
        self.client = reverb.Client(f"{ip}:{buffer_server_port}")

    def reset(self):
        self.writer = self.client.trajectory_writer(num_keep_alive_refs=self.timesteps)

    def _store_transition(self, last_observation, action, reward, terminated, *args, priority=None, truncated=False):
        """Stores a transition when in training mode.

        Stores the following tuple in the replay buffer (last_observation, action,
        reward, is_terminal, priority). Follows the dopamine replay buffer naming.

        Args:
          last_observation: Last observation
          action: An integer, the action taken after the last observation.
          reward: A float, the reward received for the action after the last observation.
          terminated: Boolean indicating if the current state is a terminal state.
          Or terminal in dopamine. Similar to gymnasium terminated.
          *args: Any, other items to be added to the replay buffer.
          priority: Float. Priority of sampling the transition. If None, the default
            priority will be used. If replay scheme is uniform, the default priority
            is 1. If the replay scheme is prioritized, the default priority is the
            maximum ever seen [Schaul et al., 2015].
          truncated: bool, whether this transition is the last for the episode.
            This can be different than terminal when ending the episode because of a
            timeout. Episode_end in dopamine. Similar to gymnasium truncated.
        """
        self.writer.append(
            {
                "observation": np.array(last_observation, dtype=np.float32),
                "action": np.array(action, dtype=np.int32),
                "reward": np.array(reward, dtype=np.float32),
                "terminated": np.array(terminated, dtype=np.float32),
            }
        )
        if self.writer.episode_steps >= self.timesteps:
            self.writer.create_item(
                table=self.table_name,
                priority=1,
                trajectory={
                    "observation": self.writer.history["observation"][-self.timesteps :],
                    "action": self.writer.history["action"][-self.timesteps :],
                    "reward": self.writer.history["reward"][-self.timesteps :],
                    "terminated": self.writer.history["terminated"][-self.timesteps :],
                },
            )
            self.writer.flush(block_until_num_items=self.timesteps)
        if terminated or truncated:
            # Block until all pending items have been sent to the server and
            # inserted into 'my_table'. This also clears the buffers so history will
            # once again be empty and `writer.episode_steps` is 0.
            self.writer.end_episode()

    def close(self):
        self.writer.close()


class HandlerReverbSampler:
    def __init__(
        self,
        gamma: float,
        batch_size: int,
        timesteps: int,
        table_name: str,
        ip: str = "localhost",
        buffer_server_port: int = 8023,
    ):
        """
        Args:
            timesteps
            table_name
            ip and buffer_server_port: this is server adress.
        """
        # dataset creation calls tf, which occupies all memory
        tf.config.experimental.set_visible_devices([], "GPU")
        self.client = reverb.Client(f"{ip}:{buffer_server_port}")
        self._cumulative_discount_vector = np.array(
            [math.pow(gamma, n) for n in range(timesteps - 1)],
            dtype=np.float32,
        )
        self.timesteps = timesteps
        self.table_name = table_name
        ds = reverb.TrajectoryDataset.from_table_signature(
            server_address=f"{ip}:{buffer_server_port}",
            table=table_name,
            max_in_flight_samples_per_worker=3 * batch_size,
        )
        ds = ds.batch(batch_size).prefetch(1)
        self.iterator = ds.as_numpy_iterator()

    def sample_from_replay_buffer(self):
        start = time.perf_counter()
        info, data = next(self.iterator)
        fetch_time = time.perf_counter() - start
        return {
            "state": data["observation"][:, 0, ...],
            "action": data["action"][:, 0],
            "reward": np.sum(self._cumulative_discount_vector * data["reward"][:, :-1], axis=1),
            "next_state": data["observation"][:, -1, ...],
            "next_action": data["action"][:, -1],
            "next_reward": data["reward"][:, -1],
            "terminal": data["terminated"][:, -1],
        }, fetch_time

    def add_count(self):
        table_info = self.client.server_info()[self.table_name]
        return table_info.current_size
    
    def checkpoint(self):
        return self.client.checkpoint()


def timeit(func):
    """Decorator to measure and report the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        print(f"Function '{func.__name__}' took {execution_time:.4f} seconds to complete.")
        return result

    return wrapper


def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps, epsilon):
    return epsilon


@functools.partial(jax.jit, static_argnums=(0, 2, 3))
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.

    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
        Begin at 1. until warmup_steps steps have been taken; then
        Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
        Use epsilon from there on.

    Args:
        decay_period: float, the period over which epsilon is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before epsilon is decayed.
        epsilon: float, the final value to which to decay the epsilon parameter.

    Returns:
        A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = jnp.clip(bonus, 0.0, 1.0 - epsilon)
    return epsilon + bonus


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11))
def select_action(
    network_def,
    params,
    state,
    rng,
    num_actions,
    eval_mode,
    epsilon_eval,
    epsilon_train,
    epsilon_decay_period,
    step,
    warmup_steps,
    epsilon_fn,
):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Args:
        network_def: Linen Module to use for inference.
        params: Linen params (frozen dict) to use for inference.
        state: input state to use for inference.
        rng: Jax random number generator.
        num_actions: int, number of actions (static_argnum).
        eval_mode: bool, whether we are in eval mode (static_argnum).
        epsilon_eval: float, epsilon value to use in eval mode (static_argnum).
        epsilon_train: float, epsilon value to use in train mode (static_argnum).
        epsilon_decay_period: float, decay period for epsilon value for certain
            epsilon functions, such as linearly_decaying_epsilon, (static_argnum).
        training_steps: int, number of training steps so far.
        min_replay_history: int, minimum number of steps in replay buffer
            (static_argnum).
        epsilon_fn: function used to calculate epsilon value (static_argnum).

    Returns:
        rng: Jax random number generator.
        action: int, the selected action.
    """
    epsilon = jnp.where(
        eval_mode,
        epsilon_eval,
        epsilon_fn(
            epsilon_decay_period,
            step,
            warmup_steps,
            epsilon_train,
        ),
    )

    state = jnp.expand_dims(state, axis=0)
    rng, rng1, rng2 = jax.random.split(rng, num=3)
    p = jax.random.uniform(rng1)
    return (
        rng,
        jnp.where(
            p <= epsilon,
            jax.random.randint(rng2, (), 0, num_actions),
            jnp.argmax(network_def.apply(params, state).q_values),
        ),
        epsilon,
    )


def restore_dqn_flax_state(num_actions, observation_shape, learning_rate, eps, checkpointdir):
    orbax_checkpointer = ocp.StandardCheckpointer()
    args_network = {"num_actions": num_actions}
    network = networks.NatureDQNNetwork
    optimizer = optax.adam
    args_optimizer = {"learning_rate": learning_rate, "eps": eps}
    rng = jax.random.PRNGKey(0)  # jax.random.key(0)
    state = dqn.create_train_state(rng, network, args_network, optimizer, args_optimizer, observation_shape)
    abstract_my_tree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    return orbax_checkpointer.restore(checkpointdir, args=ocp.args.StandardRestore(abstract_my_tree))
