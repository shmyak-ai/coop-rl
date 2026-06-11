# Copyright 2025 The Coop RL Authors.
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
#
# DreamerV3 neural-network primitives. Flax ports of the ninjax modules from the
# original implementation (embodied.jax.nets). The arithmetic (initializer
# scales, block einsum, the manual transposed convolution, RMS/Layer norm) is
# preserved unchanged; only the state-management layer (ninjax -> flax.linen)
# differs.

import math
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from coop_rl.networks.utils import parse_activation_fn

COMPUTE_DTYPE = jnp.bfloat16
f32 = jnp.float32


def cast(xs, force=False):
    def should(x):
        return True if force else jnp.issubdtype(x.dtype, jnp.floating)

    return jax.tree.map(lambda x: COMPUTE_DTYPE(x) if should(x) else x, xs)


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def get_act(name):
    if name == "none":
        return lambda x: x
    if name == "mish":
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    if name == "relu2":
        return lambda x: jnp.square(jax.nn.relu(x))
    return parse_activation_fn(name)


def where(condition, xs, ys):
    assert condition.dtype == bool, condition.dtype

    def fn(x, y):
        assert x.shape == y.shape, (x.shape, y.shape)
        expanded = jnp.expand_dims(condition, list(range(condition.ndim, x.ndim)))
        return jnp.where(expanded, x, y)

    return jax.tree.map(fn, xs, ys)


def mask(xs, m):
    return where(m, xs, jax.tree.map(jnp.zeros_like, xs))


def available(*trees, bdims=None):
    def fn(*xs):
        masks = []
        for x in xs:
            if jnp.issubdtype(x.dtype, jnp.floating):
                m = x != -jnp.inf
            elif jnp.issubdtype(x.dtype, jnp.signedinteger):
                m = x != -1
            elif jnp.issubdtype(x.dtype, jnp.unsignedinteger) or jnp.issubdtype(x.dtype, bool):
                shape = x.shape if bdims is None else x.shape[:bdims]
                m = jnp.full(shape, True, bool)
            else:
                raise NotImplementedError(x.dtype)
            if bdims is not None:
                m = m.all(tuple(range(bdims, m.ndim)))
            masks.append(m)
        return jnp.stack(masks, 0).all(0)

    return jax.tree.map(fn, *trees)


class Initializer:
    """Flax-compatible parameter initializer matching the DreamerV3 scales."""

    def __init__(self, dist="trunc_normal", fan="in", scale=1.0):
        self.dist = dist
        self.fan = fan
        self.scale = scale

    def __call__(self, key, shape, dtype=f32, fshape=None):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        assert all(isinstance(x, int) for x in shape), shape
        assert all(x > 0 for x in shape), shape
        fanin, fanout = self.compute_fans(shape if fshape is None else fshape)
        fan = {"avg": (fanin + fanout) / 2, "in": fanin, "out": fanout, "none": 1}[self.fan]
        if self.dist == "zeros":
            x = jnp.zeros(shape, dtype)
        elif self.dist == "uniform":
            limit = np.sqrt(1 / fan)
            x = jax.random.uniform(key, shape, dtype, -limit, limit)
        elif self.dist == "normal":
            x = jax.random.normal(key, shape)
            x *= np.sqrt(1 / fan)
        elif self.dist == "trunc_normal":
            x = jax.random.truncated_normal(key, -2, 2, shape)
            x *= 1.1368 * np.sqrt(1 / fan)
        elif self.dist == "normed":
            x = jax.random.uniform(key, shape, dtype, -1, 1)
            x *= 1 / jnp.linalg.norm(x.reshape((-1, shape[-1])), 2, 0)
        else:
            raise NotImplementedError(self.dist)
        x *= self.scale
        return x.astype(dtype)

    def __repr__(self):
        return f"Initializer({self.dist}, {self.fan}, {self.scale})"

    def __eq__(self, other):
        attrs = ("dist", "fan", "scale")
        return isinstance(other, Initializer) and all(
            getattr(self, k) == getattr(other, k) for k in attrs
        )

    def __hash__(self):
        return hash((self.dist, self.fan, self.scale))

    @staticmethod
    def compute_fans(shape):
        if len(shape) == 0:
            return (1, 1)
        elif len(shape) == 1:
            return (1, shape[0])
        elif len(shape) == 2:
            return shape
        else:
            space = math.prod(shape[:-2])
            return (shape[-2] * space, shape[-1] * space)


def _scaled(init, outscale):
    if outscale == 1.0:
        return init

    def fn(key, shape, dtype=f32):
        return init(key, shape, dtype) * outscale

    return fn


class Linear(nn.Module):
    units: int | tuple[int, ...]
    bias: bool = True
    winit: Any = Initializer("trunc_normal")
    binit: Any = Initializer("zeros")
    outscale: float = 1.0

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        units = (self.units,) if isinstance(self.units, int) else tuple(self.units)
        size = math.prod(units)
        kernel = self.param("kernel", _scaled(self.winit, self.outscale), (x.shape[-1], size), f32)
        x = x @ kernel.astype(x.dtype)
        if self.bias:
            x += self.param("bias", self.binit, (size,), f32).astype(x.dtype)
        x = x.reshape((*x.shape[:-1], *units))
        return x


class BlockLinear(nn.Module):
    units: int
    blocks: int
    bias: bool = True
    winit: Any = Initializer("trunc_normal")
    binit: Any = Initializer("zeros")
    outscale: float = 1.0

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        assert x.shape[-1] % self.blocks == 0, (x.shape, self.blocks)
        insize = x.shape[-1]
        shape = (self.blocks, insize // self.blocks, self.units // self.blocks)
        kernel = self.param("kernel", _scaled(self.winit, self.outscale), shape, f32).astype(
            x.dtype
        )
        x = x.reshape((*x.shape[:-1], self.blocks, insize // self.blocks))
        x = jnp.einsum("...ki,kio->...ko", x, kernel)
        x = x.reshape((*x.shape[:-2], self.units))
        if self.bias:
            x += self.param("bias", self.binit, (self.units,), f32).astype(x.dtype)
        return x


class Conv2D(nn.Module):
    depth: int
    kernel: int | tuple[int, ...]
    stride: int = 1
    transp: bool = False
    groups: int = 1
    pad: str = "same"
    bias: bool = True
    winit: Any = Initializer("trunc_normal")
    binit: Any = Initializer("zeros")
    outscale: float = 1.0

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        kern = (self.kernel, self.kernel) if isinstance(self.kernel, int) else tuple(self.kernel)
        shape = (*kern, x.shape[-1] // self.groups, self.depth)
        kernel = self.param("kernel", _scaled(self.winit, self.outscale), shape, f32).astype(
            x.dtype
        )
        if self.transp:
            assert self.pad == "same", self.pad
            # Manual fractionally strided convolution (the cuDNN path used by XLA
            # has bugs and performance issues).
            x = x.repeat(self.stride, -2).repeat(self.stride, -3)
            maskh = ((jnp.arange(x.shape[-3]) - 1) % self.stride == 0)[:, None]
            maskw = ((jnp.arange(x.shape[-2]) - 1) % self.stride == 0)[None, :]
            x *= (maskh * maskw)[:, :, None]
            stride = (1, 1)
        else:
            stride = (self.stride, self.stride)
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            stride,
            self.pad.upper(),
            feature_group_count=self.groups,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias:
            x += self.param("bias", self.binit, (self.depth,), f32).astype(x.dtype)
        return x


class Norm(nn.Module):
    impl: str = "rms"
    eps: float = 1e-4
    scale: bool = True
    shift: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        impl, eps = self.impl, self.eps
        if "1em" in impl:
            impl, exp = impl.split("1em")
            eps = 10 ** -int(exp)
        dtype = x.dtype
        x = f32(x)
        shape = (x.shape[-1],)
        if impl == "none":
            return x.astype(dtype)
        elif impl == "rms":
            mean2 = jnp.square(x).mean(-1, keepdims=True)
            scale = self._scale(shape)
            x = x * (jax.lax.rsqrt(mean2 + eps) * scale)
        elif impl == "layer":
            mean = x.mean(-1, keepdims=True)
            mean2 = jnp.square(x).mean(-1, keepdims=True)
            var = jnp.maximum(0, mean2 - jnp.square(mean))
            scale = self._scale(shape)
            shift = self._shift(shape)
            x = (x - mean) * (jax.lax.rsqrt(var + eps) * scale) + shift
        else:
            raise NotImplementedError(impl)
        return x.astype(dtype)

    def _scale(self, shape):
        if not self.scale:
            return jnp.ones(shape, f32)
        return self.param("scale", nn.initializers.ones, shape, f32)

    def _shift(self, shape):
        if not self.shift:
            return jnp.zeros(shape, f32)
        return self.param("shift", nn.initializers.zeros, shape, f32)


class MLP(nn.Module):
    layers: int = 5
    units: int = 1024
    act: str = "silu"
    norm: str = "rms"
    bias: bool = True
    winit: Any = Initializer("trunc_normal")
    binit: Any = Initializer("zeros")

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        shape = x.shape[:-1]
        x = x.astype(COMPUTE_DTYPE)
        x = x.reshape([-1, x.shape[-1]])
        kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
        for i in range(self.layers):
            x = Linear(self.units, **kw, name=f"linear{i}")(x)
            x = Norm(self.norm, name=f"norm{i}")(x)
            x = get_act(self.act)(x)
        x = x.reshape((*shape, x.shape[-1]))
        return x


class DictConcat:
    """Concatenate a dict of (vector / discrete) inputs into one tensor.

    Faithful port of the ninjax DictConcat for the non-image path (used to embed
    the action dict before the RSSM core). Images are not supported here.
    """

    def __init__(self, spaces, fdims, squish=lambda x: x):
        assert fdims >= 1, fdims
        self.keys = sorted(spaces.keys())
        self.spaces = spaces
        self.fdims = fdims
        self.squish = squish

    def __call__(self, xs):
        assert all(k in xs for k in self.spaces), (self.spaces, xs.keys())
        bdims = xs[self.keys[0]].ndim - len(self.spaces[self.keys[0]].shape)
        ys = []
        for key in self.keys:
            space = self.spaces[key]
            x = xs[key]
            m = available(x, bdims=bdims)
            x = mask(x, m)
            assert x.shape[bdims:] == space.shape, (key, bdims, space.shape, x.shape)
            if space.dtype == jnp.uint8 and len(space.shape) in (2, 3):
                raise NotImplementedError("Images are not supported.")
            elif space.discrete:
                classes = np.asarray(space.classes).flatten()
                assert (classes == classes[0]).all(), classes
                classes = classes[0].item()
                x = x.astype(jnp.int32)
                x = jax.nn.one_hot(x, classes, dtype=COMPUTE_DTYPE)
            else:
                x = self.squish(x)
                x = x.astype(COMPUTE_DTYPE)
            x = mask(x, m)
            x = x.reshape((*x.shape[: bdims + self.fdims - 1], -1))
            ys.append(x)
        return jnp.concatenate(ys, -1)
