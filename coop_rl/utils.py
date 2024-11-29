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

import jax
import jax.numpy as jnp


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
