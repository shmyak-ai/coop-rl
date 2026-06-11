#
# from Stoix https://github.com/EdanToledo/Stoix
#

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from coop_rl.base import distributions as outs
from coop_rl.networks.dreamer_nn import MLP, Linear, symexp, symlog
from coop_rl.networks.epsilon_greedy import EpsilonGreedy

f32 = jnp.float32


class ScalarCriticHead(nn.Module):
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return nn.Dense(1, kernel_init=self.kernel_init)(embedding).squeeze(axis=-1)


class DiscreteQNetworkHead(nn.Module):
    action_dim: int
    epsilon: float = 0.1
    kernel_init: Initializer = orthogonal(1.0)
    dtype: Any = None

    @nn.compact
    def __call__(self, embedding: chex.Array) -> EpsilonGreedy:
        q_values = nn.Dense(self.action_dim, kernel_init=self.kernel_init, dtype=self.dtype)(
            embedding
        )
        return EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)


class LinearHead(nn.Module):
    output_dim: int
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return nn.Dense(self.output_dim, kernel_init=self.kernel_init)(embedding)


# ===========================================================================
# DreamerV3 output heads. Flax ports of the ninjax embodied.jax.heads. A Head
# maps an MLP embedding to a distribution Output; MLPHead / DictMLPHead bundle
# an MLP trunk with a single / per-key Head respectively. The arithmetic
# (twohot bins, straight-through onehot, bounded normal) is unchanged.
# ===========================================================================


class HeadSpec(NamedTuple):
    impl: str
    shape: tuple[int, ...] = ()
    classes: int = 0
    bins: int = 255
    unimix: float = 0.0
    minstd: float = 1.0
    maxstd: float = 1.0
    outscale: float = 1.0


def _twohot_bins(bins: int) -> chex.Array:
    if bins % 2 == 1:
        half = symexp(jnp.linspace(-20, 0, (bins - 1) // 2 + 1, dtype=f32))
        return jnp.concatenate([half, -half[:-1][::-1]], 0)
    half = symexp(jnp.linspace(-20, 0, bins // 2, dtype=f32))
    return jnp.concatenate([half, -half[::-1]], 0)


class Head(nn.Module):
    impl: str
    shape: tuple[int, ...] = ()
    classes: int = 0
    bins: int = 255
    unimix: float = 0.0
    minstd: float = 1.0
    maxstd: float = 1.0
    outscale: float = 1.0

    @nn.compact
    def __call__(self, x: chex.Array):
        os = self.outscale
        impl = self.impl
        if impl == "binary":
            out = outs.Binary(Linear(self.shape, outscale=os, name="logit")(x))
        elif impl == "categorical":
            logits = Linear((*self.shape, self.classes), outscale=os, name="logits")(x)
            out = outs.Categorical(logits)
            out.minent = 0
            out.maxent = float(np.log(self.classes))
        elif impl == "onehot":
            logits = Linear(self.shape, outscale=os, name="logits")(x)
            out = outs.OneHot(logits, self.unimix)
        elif impl == "mse":
            out = outs.MSE(Linear(self.shape, outscale=os, name="pred")(x))
        elif impl == "symlog_mse":
            out = outs.MSE(Linear(self.shape, outscale=os, name="pred")(x), symlog)
        elif impl == "symexp_twohot":
            logits = Linear((*self.shape, self.bins), outscale=os, name="logits")(x)
            out = outs.TwoHot(logits, _twohot_bins(self.bins))
        elif impl == "bounded_normal":
            mean = Linear(self.shape, outscale=os, name="mean")(x)
            stddev = Linear(self.shape, outscale=os, name="stddev")(x)
            lo, hi = self.minstd, self.maxstd
            stddev = (hi - lo) * jax.nn.sigmoid(stddev + 2.0) + lo
            out = outs.Normal(jnp.tanh(mean), stddev)
            out.minent = outs.Normal(jnp.zeros_like(mean), self.minstd).entropy()
            out.maxent = outs.Normal(jnp.zeros_like(mean), self.maxstd).entropy()
        else:
            raise NotImplementedError(impl)
        if self.shape:
            out = outs.Agg(out, len(self.shape), jnp.sum)
        return out


class MLPHead(nn.Module):
    layers: int
    units: int
    act: str
    norm: str
    spec: HeadSpec

    @nn.compact
    def __call__(self, x: chex.Array, bdims: int):
        bshape = x.shape[:bdims]
        x = x.reshape((*bshape, -1))
        x = MLP(self.layers, self.units, self.act, self.norm, name="mlp")(x)
        return Head(*self.spec, name="head")(x)


class DictMLPHead(nn.Module):
    layers: int
    units: int
    act: str
    norm: str
    keys: tuple[str, ...]
    specs: tuple[HeadSpec, ...]

    @nn.compact
    def __call__(self, x: chex.Array, bdims: int):
        bshape = x.shape[:bdims]
        x = x.reshape((*bshape, -1))
        x = MLP(self.layers, self.units, self.act, self.norm, name="mlp")(x)
        return {k: Head(*spec, name=k)(x) for k, spec in zip(self.keys, self.specs, strict=True)}
