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
import time

import gymnasium as gym
import jax
import jax.numpy as jnp
from gymnasium.wrappers import FrameStack


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
          action: An integer, the action taken.
          reward: A float, the reward.
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
            last_observation = last_observation[-1, ...]

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
    training_steps,
    min_replay_history,
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
            training_steps,
            min_replay_history,
            epsilon_train,
        ),
    )

    rng, rng1, rng2 = jax.random.split(rng, num=3)
    p = jax.random.uniform(rng1)
    return rng, jnp.where(
        p <= epsilon,
        jax.random.randint(rng2, (), 0, num_actions),
        jnp.argmax(network_def.apply(params, state).q_values),
    )
