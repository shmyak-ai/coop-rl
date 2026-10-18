#
# An ensemble of independently initialized quantile networks (Osband et al. 2016's bootstrapped
# DQN without the masks): K members evaluated in one vectorized call.
#

from collections.abc import Callable, Mapping
from typing import Any

import chex
import jax.numpy as jnp
from flax import linen as nn

from coop_rl.networks.epsilon_greedy import EpsilonGreedy


def mean_preferences(z_values: chex.Array, quantiles: chex.Array) -> chex.Array:
    """The members' mean q-value: z (K, ..., N, A) -> (..., A)."""
    del quantiles
    return z_values.mean(axis=(0, -2))


class _Member(nn.Module):
    network: type[nn.Module]
    args_network: Mapping[str, Any]
    num_quantiles: int  # a field, not an argument: vmap would trace it, and it is a shape

    @nn.compact
    def __call__(self, observation):
        _, z_values, quantiles = self.network(**self.args_network)(observation, self.num_quantiles)
        return z_values, quantiles


class EnsembleQuantileNetwork(nn.Module):
    """K copies of a quantile network (e.g. QuantileFeedForwardNetwork), each with its own
    parameters and NoisyNet noise, returning every member's samples on a leading member axis.

    By default the members share each call's quantile fractions, so their spread is the
    parameters' alone, not tau-sampling noise; `split_quantiles=True` (for the loss) draws
    their own. `combine_fn(z (K, ..., N, A), quantiles (K, ..., N))` turns the members into
    the acting policy's preferences; the loss trains each member on its own (the update
    step's `ensemble*` options). The members' own heads' `cvar_eta` and policies are unused:
    risk-sensitive acting belongs in `combine_fn`.
    """

    network: type[nn.Module]
    args_network: Mapping[str, Any]
    num_members: int
    epsilon: float
    combine_fn: Callable[[chex.Array, chex.Array], chex.Array] = mean_preferences

    @nn.compact
    def __call__(
        self,
        observation,
        num_quantiles: int,
        epsilon: float | None = None,
        split_quantiles: bool = False,
    ) -> tuple[EpsilonGreedy, chex.Array, chex.Array]:
        """Returns (eps-greedy policy over combine_fn, z (K, ..., N, A), quantiles (K, ..., N))."""
        members = nn.vmap(
            _Member,
            variable_axes={"params": 0},
            split_rngs={"params": True, "noise": True, "quantiles": split_quantiles},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_members,
        )(self.network, self.args_network, num_quantiles, name="members")
        z_values, quantiles = members(observation)
        preferences = self.combine_fn(z_values.astype(jnp.float32), quantiles)
        eps = self.epsilon if epsilon is None else epsilon
        return EpsilonGreedy(preferences=preferences, epsilon=eps), z_values, quantiles
