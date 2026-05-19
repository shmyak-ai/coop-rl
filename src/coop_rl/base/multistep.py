#
# from Stoix https://github.com/EdanToledo/Stoix
#

import chex
import jax
import jax.numpy as jnp

# These functions are generally taken from rlax but edited to explicitly take in a batch of data.
# This is because the original rlax functions are not batched and are meant to be used with vmap,
# which can be much slower.


def batch_lambda_returns(
    r_t: chex.Array,
    discount_t: chex.Array,
    v_t: chex.Array,
    lambda_: chex.Numeric = 1.0,
    stop_target_gradients: bool = False,
    time_major: bool = False,
) -> chex.Array:
    """Estimates a multistep truncated lambda return from a trajectory.

    Given a a trajectory of length `T+1`, generated under some policy π, for each
    time-step `t` we can estimate a target return `G_t`, by combining rewards,
    discounts, and state values, according to a mixing parameter `lambda`.

    The parameter `lambda_`  mixes the different multi-step bootstrapped returns,
    corresponding to accumulating `k` rewards and then bootstrapping using `v_t`.

        rₜ₊₁ + γₜ₊₁ vₜ₊₁
        rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ vₜ₊₂
        rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ rₜ₊₂ + γₜ₊₁ γₜ₊₂ γₜ₊₃ vₜ₊₃

    The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:

        Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].

    In the `on-policy` case, we estimate a return target `G_t` for the same
    policy π that was used to generate the trajectory. In this setting the
    parameter `lambda_` is typically a fixed scalar factor. Depending
    on how values `v_t` are computed, this function can be used to construct
    targets for different multistep reinforcement learning updates:

        TD(λ):  `v_t` contains the state value estimates for each state under π.
        Q(λ):  `v_t = max(q_t, axis=-1)`, where `q_t` estimates the action values.
        Sarsa(λ):  `v_t = q_t[..., a_t]`, where `q_t` estimates the action values.

    In the `off-policy` case, the mixing factor is a function of state, and
    different definitions of `lambda` implement different off-policy corrections:

        Per-decision importance sampling:  λₜ = λ ρₜ = λ [π(aₜ|sₜ) / μ(aₜ|sₜ)]
        V-trace, as instantiated in IMPALA:  λₜ = min(1, ρₜ)

    Note that the second option is equivalent to applying per-decision importance
    sampling, but using an adaptive λ(ρₜ) = min(1/ρₜ, 1), such that the effective
    bootstrap parameter at time t becomes λₜ = λ(ρₜ) * ρₜ = min(1, ρₜ).
    This is the interpretation used in the ABQ(ζ) algorithm (Mahmood 2017).

    Of course this can be augmented to include an additional factor λ.  For
    instance we could use V-trace with a fixed additional parameter λ = 0.9, by
    setting λₜ = 0.9 * min(1, ρₜ) or, alternatively (but not equivalently),
    λₜ = min(0.9, ρₜ).

    Estimated return are then often used to define a td error, e.g.:  ρₜ(Gₜ - vₜ).

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/sutton/book/ebook/node74.html).

    Args:
        r_t: sequence of rewards rₜ for timesteps t in B x [1, T].
        discount_t: sequence of discounts γₜ for timesteps t in B x [1, T].
        v_t: sequence of state values estimates under π for timesteps t in B x [1, T].
        lambda_: mixing parameter; a scalar or a vector for timesteps t in B x [1, T].
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
        time_major: If True, the first dimension of the input tensors is the time
        dimension.

    Returns:
        Multistep lambda returns.
    """

    chex.assert_rank([r_t, discount_t, v_t, lambda_], [2, 2, 2, {0, 1, 2}])
    chex.assert_type([r_t, discount_t, v_t, lambda_], float)
    chex.assert_equal_shape([r_t, discount_t, v_t])

    # Swap axes to make time axis the first dimension
    if not time_major:
        r_t, discount_t, v_t = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (r_t, discount_t, v_t)
        )

    # If scalar make into vector.
    lambda_ = jnp.ones_like(discount_t) * lambda_

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    def _body(
        acc: chex.Array, xs: tuple[chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> tuple[chex.Array, chex.Array]:
        returns, discounts, values, lambda_ = xs
        acc = returns + discounts * ((1 - lambda_) * values + lambda_ * acc)
        return acc, acc

    _, returns = jax.lax.scan(_body, v_t[-1], (r_t, discount_t, v_t, lambda_), reverse=True)

    if not time_major:
        returns = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), returns)

    return jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(returns), returns)


def batch_discounted_returns(
    r_t: chex.Array,
    discount_t: chex.Array,
    v_t: chex.Array,
    stop_target_gradients: bool = False,
    time_major: bool = False,
) -> chex.Array:
    """Calculates a discounted return from a trajectory.

    The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:

        Gₜ = rₜ₊₁ + γₜ₊₁ Gₜ₊₁.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/sutton/book/ebook/node61.html).

    Args:
        r_t: reward sequence at time t.
        discount_t: discount sequence at time t.
        v_t: value sequence or scalar at time t.
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
        Discounted returns.
    """
    chex.assert_rank([r_t, discount_t, v_t], [2, 2, {0, 1, 2}])
    chex.assert_type([r_t, discount_t, v_t], float)

    # If scalar make into vector.
    bootstrapped_v = jnp.ones_like(discount_t) * v_t
    return batch_lambda_returns(
        r_t,
        discount_t,
        bootstrapped_v,
        lambda_=1.0,
        stop_target_gradients=stop_target_gradients,
        time_major=time_major,
    )
