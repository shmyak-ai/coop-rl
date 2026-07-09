#
# from Stoix https://github.com/EdanToledo/Stoix
#

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from coop_rl.base.base_types import Observation, RNNObservation
from coop_rl.networks.inputs import ObservationInput
from coop_rl.networks.utils import parse_rnn_cell


class FeedForwardNetwork(nn.Module):
    """Simple Feedforward Network."""

    torso: type[nn.Module]
    args_torso: Mapping[str, Any]
    head: type[nn.Module]
    args_head: Mapping[str, Any]
    input_layer: type[nn.Module] = ObservationInput

    @nn.compact
    def __call__(self, observation: Observation) -> Any:
        x = self.input_layer()(observation)
        x = self.torso(**self.args_torso)(x)
        return self.head(**self.args_head)(x)


class QuantileFeedForwardNetwork(nn.Module):
    """Feedforward network whose head takes a number of quantile samples (IQN-style)."""

    torso: type[nn.Module]
    args_torso: Mapping[str, Any]
    head: type[nn.Module]
    args_head: Mapping[str, Any]
    input_layer: type[nn.Module] = ObservationInput

    @nn.compact
    def __call__(self, observation: Observation, num_quantiles: int) -> Any:
        x = self.input_layer()(observation)
        x = self.torso(**self.args_torso)(x)
        return self.head(**self.args_head)(x, num_quantiles)


class CompositeNetwork(nn.Module):
    """Composite Network. Takes in a sequence of layers and applies them sequentially."""

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, *network_input: chex.Array | tuple[chex.Array, ...]) -> Any | chex.Array:
        x = self.layers[0](*network_input)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class MultiNetwork(nn.Module):
    """Multi Network.

    Takes in a sequence of networks, applies them separately and concatenates the outputs."""

    networks: Sequence[nn.Module]

    @nn.compact
    def __call__(self, *network_input: chex.Array | tuple[chex.Array, ...]) -> Any | chex.Array:
        """Forward pass."""
        outputs = []
        for network in self.networks:
            outputs.append(network(*network_input))
        concatenated = jnp.stack(outputs, axis=-1)
        chex.assert_rank(concatenated, 2)
        return concatenated


class ScannedRNN(nn.Module):
    hidden_state_dim: int
    cell_type: str

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, rnn_state: chex.Array, x: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Applies the module."""
        ins, resets = x

        def hidden_state_reset_fn(reset_state, current_state):
            return jnp.where(resets[:, np.newaxis], reset_state, current_state)

        rnn_state = jax.tree_util.tree_map(
            hidden_state_reset_fn,
            self.initialize_carry(ins.shape[0]),
            rnn_state,
        )
        new_rnn_state, y = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)(
            rnn_state, ins
        )
        return new_rnn_state, y

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_state_dim))


class RecurrentNetwork(nn.Module):
    """Recurrent Network."""

    pre_torso: type[nn.Module]
    args_pre_torso: Mapping[str, Any]
    post_torso: type[nn.Module]
    args_post_torso: Mapping[str, Any]
    head: type[nn.Module]
    args_head: Mapping[str, Any]
    hidden_state_dim: int
    cell_type: str
    action_dim: int
    input_layer: type[nn.Module] = ObservationInput

    @nn.compact
    def __call__(
        self,
        hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> tuple[chex.Array, Any]:
        x, prev_action, prev_reward, done = observation_done

        x = self.input_layer()(x)
        x = self.pre_torso(**self.args_pre_torso)(x)
        extras = jnp.concatenate(
            [jax.nn.one_hot(prev_action, self.action_dim), prev_reward[..., jnp.newaxis]],
            axis=-1,
        ).astype(x.dtype)
        x = jnp.concatenate([x, extras], axis=-1)
        rnn_input = (x, done)
        hidden_state, x = ScannedRNN(self.hidden_state_dim, self.cell_type)(hidden_state, rnn_input)
        x = self.post_torso(**self.args_post_torso)(x)
        x = self.head(**self.args_head)(x)

        return hidden_state, x


class QuantileRecurrentNetwork(nn.Module):
    """Recurrent network whose head takes a number of quantile samples (IQN-style)."""

    pre_torso: type[nn.Module]
    args_pre_torso: Mapping[str, Any]
    post_torso: type[nn.Module]
    args_post_torso: Mapping[str, Any]
    head: type[nn.Module]
    args_head: Mapping[str, Any]
    hidden_state_dim: int
    cell_type: str
    action_dim: int
    input_layer: type[nn.Module] = ObservationInput

    @nn.compact
    def __call__(
        self,
        hidden_state: chex.Array,
        observation_done: RNNObservation,
        num_quantiles: int,
    ) -> tuple[chex.Array, Any]:
        x, prev_action, prev_reward, done = observation_done

        x = self.input_layer()(x)
        x = self.pre_torso(**self.args_pre_torso)(x)
        extras = jnp.concatenate(
            [jax.nn.one_hot(prev_action, self.action_dim), prev_reward[..., jnp.newaxis]],
            axis=-1,
        ).astype(x.dtype)
        x = jnp.concatenate([x, extras], axis=-1)
        rnn_input = (x, done)
        hidden_state, x = ScannedRNN(self.hidden_state_dim, self.cell_type)(hidden_state, rnn_input)
        x = self.post_torso(**self.args_post_torso)(x)
        x = self.head(**self.args_head)(x, num_quantiles)

        return hidden_state, x
