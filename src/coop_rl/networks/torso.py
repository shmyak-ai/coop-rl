#
# from Stoix https://github.com/EdanToledo/Stoix
#

from collections.abc import Sequence
from typing import Any

import chex
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal, variance_scaling

from coop_rl.networks.layers import NoisyLinear
from coop_rl.networks.utils import parse_activation_fn

# LeCun uniform initializer used in Wang et al. (NeurIPS 2025) deep residual networks.
_lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")


def _residual_block(x: chex.Array, width: int, normalize, activation, dtype: Any) -> chex.Array:
    """One residual block: 4× (Dense → LayerNorm → activation) + additive skip."""
    identity = x
    for _ in range(4):
        x = nn.Dense(width, kernel_init=_lecun_uniform, use_bias=False, dtype=dtype)(x)
        x = normalize(x)
        x = activation(x)
    return x + identity


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True
    dtype: Any = None

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size,
                kernel_init=self.kernel_init,
                use_bias=not self.use_layer_norm,
                dtype=self.dtype,
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class NoisyMLPTorso(nn.Module):
    """MLP torso using NoisyLinear layers instead of standard Dense layers."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True
    sigma_zero: float = 0.5
    dtype: Any = None

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        x = observation
        for layer_size in self.layer_sizes:
            x = NoisyLinear(
                layer_size,
                sigma_zero=self.sigma_zero,
                use_bias=not self.use_layer_norm,
                dtype=self.dtype,
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class DeepResidualTorso(nn.Module):
    """Deep residual MLP torso from Wang et al. (NeurIPS 2025 Best Paper).

    Scales to 1000+ layers via residual connections + LayerNorm + Swish.
    All three components are jointly essential for stable depth scaling.

    Args:
        width: Hidden size of every Dense layer.
        depth: Total Dense layers; must be a multiple of 4 (4 per residual block).
        activation: Activation name. 'swish' strongly preferred; 'relu' degrades at depth.
        dtype: Dtype for Dense layers (e.g. jnp.bfloat16).
    """

    width: int = 256
    depth: int = 16
    activation: str = "swish"
    dtype: Any = None

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        assert self.depth % 4 == 0, f"depth must be a multiple of 4, got {self.depth}"
        normalize = nn.LayerNorm()
        act = parse_activation_fn(self.activation)

        x = nn.Dense(self.width, kernel_init=_lecun_uniform, use_bias=False, dtype=self.dtype)(x)
        x = normalize(x)
        x = act(x)

        for _ in range(self.depth // 4):
            x = _residual_block(x, self.width, normalize, act, self.dtype)

        return x


class CNNTorso(nn.Module):
    """2D CNN torso. Expects input of shape (batch, height, width, channels).
    After flattening, feeds into a DeepResidualTorso (Wang et al., NeurIPS 2025)."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    channel_first: bool = False
    dtype: Any = None
    width: int = 256
    depth: int = 16

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        if self.channel_first:
            x = x.transpose((0, 2, 3, 1))
        for channel, kernel, stride in zip(
            self.channel_sizes, self.kernel_sizes, self.strides, strict=True
        ):
            x = nn.Conv(
                channel,
                (kernel, kernel),
                (stride, stride),
                use_bias=not self.use_layer_norm,
                dtype=self.dtype,
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(reduction_axes=(-3, -2, -1))(x)
            x = parse_activation_fn(self.activation)(x)

        x = x.reshape(*observation.shape[:-3], -1)

        x = DeepResidualTorso(
            width=self.width,
            depth=self.depth,
            activation="swish",
            dtype=self.dtype,
        )(x)

        return x
