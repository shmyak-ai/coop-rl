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
import optax
import orbax.checkpoint as ocp

from coop_rl import networks
from coop_rl.agents import dqn


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
