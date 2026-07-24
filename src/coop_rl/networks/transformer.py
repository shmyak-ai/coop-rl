# Decoder-only causal Transformer torso (Kronos-style: RoPE, pre-LN, SwiGLU),
# ported from exchange_gym's own model/module.py (RotaryPositionalEmbedding,
# MultiHeadAttentionWithRoPE, TransformerBlock). Domain-agnostic: consumes an
# already-embedded (batch, T, d_model) sequence produced upstream by a
# task-specific input_layer, plus a (batch, T, T) boolean attention mask, and
# outputs the full sequence (not pooled), so existing per-position heads
# (e.g. NoisyDuelingQuantileQNetworkHead) apply unchanged.

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


def build_causal_boundary_mask(terminated: chex.Array, truncated: chex.Array) -> chex.Array:
    """(batch, T) terminated/truncated -> (batch, T, T) bool allow-mask.

    True at [b, i, j] iff j <= i (causal) and no episode boundary lies between
    j and i within this window. Each done=True position starts a new episode
    id for every later position (exclusive cumsum), so positions separated by
    a boundary get different ids and never attend to each other; this also
    isolates not-yet-real padding (seeded with truncated=1) from real data
    with no separate padding/validity concept needed.
    """
    done = jnp.logical_or(terminated, truncated).astype(jnp.int32)
    episode_id = jnp.cumsum(done, axis=-1) - done  # exclusive cumsum
    t = done.shape[-1]
    causal = jnp.tril(jnp.ones((t, t), dtype=bool))
    same_episode = episode_id[:, :, None] == episode_id[:, None, :]
    return causal[None] & same_episode


def _rotate_half(x: chex.Array) -> chex.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _rope_cos_sin(seq_len: int, head_dim: int) -> tuple[chex.Array, chex.Array]:
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (T, head_dim)
    return jnp.cos(emb), jnp.sin(emb)


def _apply_rope(x: chex.Array, cos: chex.Array, sin: chex.Array) -> chex.Array:
    # x: (batch, T, heads, head_dim); cos/sin: (T, head_dim)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttentionRoPE(nn.Module):
    """Multi-head self-attention with rotary position embeddings on q/k."""

    d_model: int
    num_heads: int
    dtype: Any = None

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        b, t, _ = x.shape
        head_dim = self.d_model // self.num_heads
        q = nn.Dense(self.d_model, dtype=self.dtype)(x).reshape(b, t, self.num_heads, head_dim)
        k = nn.Dense(self.d_model, dtype=self.dtype)(x).reshape(b, t, self.num_heads, head_dim)
        v = nn.Dense(self.d_model, dtype=self.dtype)(x).reshape(b, t, self.num_heads, head_dim)
        cos, sin = _rope_cos_sin(t, head_dim)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(head_dim).astype(q.dtype)
        scores = jnp.where(mask[:, None], scores.astype(jnp.float32), -1e9)
        attn = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(b, t, self.d_model)
        return nn.Dense(self.d_model, dtype=self.dtype)(out)


class _SwiGLU(nn.Module):
    d_model: int
    d_ff: int
    dtype: Any = None

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        w1 = nn.Dense(self.d_ff, use_bias=False, dtype=self.dtype)(x)
        w3 = nn.Dense(self.d_ff, use_bias=False, dtype=self.dtype)(x)
        return nn.Dense(self.d_model, use_bias=False, dtype=self.dtype)(nn.silu(w1) * w3)


class _TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dtype: Any = None

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        x = x + CausalSelfAttentionRoPE(self.d_model, self.num_heads, self.dtype)(
            nn.RMSNorm(dtype=self.dtype)(x), mask
        )
        x = x + _SwiGLU(self.d_model, self.d_ff, self.dtype)(nn.RMSNorm(dtype=self.dtype)(x))
        return x


class CausalTransformerTorso(nn.Module):
    """Stack of pre-LN causal self-attention + SwiGLU blocks (Kronos-small sizing
    by default: d_model=512, num_heads=8, num_layers=8, d_ff=1024).

    __call__(self, x) where x = (embedded, mask): embedded is (batch, T, d_model)
    from an upstream input_layer, mask is (batch, T, T) bool from
    build_causal_boundary_mask. Returns the full (batch, T, d_model) sequence.
    """

    num_layers: int = 8
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 1024
    dtype: Any = None

    @nn.compact
    def __call__(self, x: tuple[chex.Array, chex.Array]) -> chex.Array:
        embedded, mask = x
        h = embedded
        for _ in range(self.num_layers):
            h = _TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dtype)(h, mask)
        return nn.RMSNorm(dtype=self.dtype)(h)
