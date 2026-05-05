---
name: deep-residual-networks
description: "Build very deep networks (16–1024 layers) for RL. Use when: implementing deep residual MLPs for actors or critics; scaling RL network depth; adding residual blocks; preventing training instability in deep networks; improving goal-conditioned or contrastive RL performance. Based on '1000 Layer Networks for Self-Supervised RL' (NeurIPS 2025 Best Paper). Reference code: https://github.com/wang-kevin3290/scaling-crl"
argument-hint: "Describe the network role (actor/critic/encoder) and desired depth (e.g., 'critic encoder, depth 64')"
---

# Deep Residual Networks for RL

Based on **"1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities"** (Wang et al., NeurIPS 2025 Best Paper).  
Paper: [arXiv:2503.14858](https://arxiv.org/abs/2503.14858) | Code: [wang-kevin3290/scaling-crl](https://github.com/wang-kevin3290/scaling-crl)

## Key Findings

- Depth 4–1024 layers → **2×–50× performance gains** over shallow baselines on locomotion/manipulation tasks
- Depth outperforms width: a depth-8 network at width 256 beats depth-4 at width 4096
- Performance jumps emerge at **critical depths** (not smooth scaling): e.g., depth 8 for Ant Big Maze, depth 64 for Humanoid U-Maze
- All three components are **jointly essential**: residual connections + LayerNorm + Swish activation. Removing any one degrades performance significantly.
- Deeper networks unlock batch size scaling (batch size scaling gives marginal gains in shallow nets)

## The Architecture

Each residual block contains **4 Dense layers**, each followed by LayerNorm → Swish, with an additive identity skip connection:

```
input → Dense → LayerNorm → Swish
      → Dense → LayerNorm → Swish
      → Dense → LayerNorm → Swish
      → Dense → LayerNorm → Swish
      → + identity → output
```

**Depth parameter = total layers = num_residual_blocks × 4**  
(depth 16 → 4 blocks; depth 64 → 16 blocks; depth 256 → 64 blocks)

The full network is: Initial Dense → LayerNorm → Swish → [N residual blocks] → Output Dense

## Procedure

### 1. Add `DeepResidualTorso` to `networks/torso.py`

Add the following class (see [full implementation](./references/implementation.md)):

```python
from flax.linen.initializers import variance_scaling

_lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")

def _residual_block(x, width, normalize, activation, dtype):
    identity = x
    for _ in range(4):
        x = nn.Dense(width, kernel_init=_lecun_uniform, use_bias=False, dtype=dtype)(x)
        x = normalize(x)
        x = activation(x)
    return x + identity


class DeepResidualTorso(nn.Module):
    """Deep residual MLP torso from Wang et al. (NeurIPS 2025).

    depth must be a multiple of 4. Each residual block = 4 layers.
    LayerNorm is always applied (essential for stability).
    """
    width: int = 256
    depth: int = 16          # total layers; must be multiple of 4
    activation: str = "swish"
    dtype: Any = None        # propagated to all Dense layers

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        normalize = nn.LayerNorm()
        act = parse_activation_fn(self.activation)

        x = nn.Dense(self.width, kernel_init=_lecun_uniform, use_bias=False, dtype=self.dtype)(x)
        x = normalize(x)
        x = act(x)

        for _ in range(self.depth // 4):
            x = _residual_block(x, self.width, normalize, act, self.dtype)

        return x
```

### 2. Register in `networks/__init__.py`

```python
from coop_rl.networks.torso import DeepResidualTorso
```

### 3. Configure depth and width

| Setting | Recommended value |
|---------|-----------------|
| `activation` | `"swish"` (ReLU degrades at depth, per ablation) |
| LayerNorm | Always on — not configurable, essential |
| `dtype` | Match the rest of the network (e.g. `jnp.bfloat16`) |
| `width` | 256 (default; increasing width helps less than increasing depth) |
| Starting depth | 8 or 16 — step up to 32, 64, 128 |
| Actor depth limit | ≤ 512 (actor loss can explode at 1024 layers) |
| Critic depth | Up to 1024 (more robust to extreme depth) |

In a `ConfigDict`:

```python
config.network = ConfigDict()
config.network.torso = "DeepResidualTorso"
config.network.width = 256
config.network.depth = 64       # start here for most tasks
config.network.activation = "swish"
config.network.dtype = jnp.bfloat16
```

### 4. Scale batch size with depth

Batch size scaling only helps when the network is deep. A useful heuristic:

| Depth | Batch size |
|-------|-----------|
| 4–8   | 256       |
| 16–32 | 512       |
| 64+   | 1024–2048 |

### 5. Verify stability

Check these failure modes:
- **Actor loss exploding at startup**: reduce actor depth (keep critic deeper)
- **Training instability**: confirm LayerNorm is enabled and activation is `"swish"`
- **No improvement from depth**: check that residual connections are active (not zero-initialized)

See [implementation details and ablations](./references/implementation.md) for full code patterns, optional hyperspherical normalization (SimBa-v2), and parameter count estimates.
