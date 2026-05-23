---
name: deep-residual-networks
description: "Build very deep networks (16–1024 layers) for RL. Use when: implementing deep residual MLPs for actors or critics; scaling RL network depth; adding residual blocks; preventing training instability in deep networks; improving goal-conditioned or contrastive RL performance. Based on '1000 Layer Networks for Self-Supervised RL' (NeurIPS 2025 Best Paper). Reference code: https://github.com/wang-kevin3290/scaling-crl"
argument-hint: "Describe the network role (actor/critic/encoder) and desired depth (e.g., 'critic encoder, depth 64')"
---

# Deep Residual Networks for RL

Based on **"1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities"** (Wang et al., NeurIPS 2025 Best Paper).  
Paper: [arXiv:2503.14858](https://arxiv.org/abs/2503.14858) | Code: [wang-kevin3290/scaling-crl](https://github.com/wang-kevin3290/scaling-crl)

## Key Findings

- Depth 4–1024 layers → **2×–50× performance gains** over shallow baselines
- Depth outperforms width: depth-8 at width 256 beats depth-4 at width 4096
- Performance jumps at **critical depths** (not smooth): e.g., depth 8 for Ant Big Maze, depth 64 for Humanoid U-Maze
- All three components are **jointly essential**: residual connections + LayerNorm + Swish. Removing any one degrades significantly.
- Deeper networks unlock batch size scaling

## Architecture

Each residual block: 4 Dense layers, each followed by LayerNorm → Swish, with additive identity skip.

**Depth = num_residual_blocks × 4** (depth 16 → 4 blocks; depth 64 → 16 blocks)

Full network: Initial Dense → LayerNorm → Swish → [N residual blocks] → Output Dense

## Procedure

### 1. Add `DeepResidualTorso` to `networks/torso.py`

See [full implementation](./references/implementation.md) for the complete drop-in code.

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
    width: int = 256
    depth: int = 16  # must be multiple of 4
    activation: str = "swish"
    dtype: Any = None

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

### 3. Configure

| Setting | Recommended |
|---------|-------------|
| `activation` | `"swish"` (ReLU degrades at depth) |
| LayerNorm | Always on — essential, not configurable |
| `width` | 256 (depth helps more than width) |
| Starting depth | 8 or 16, step up to 32 → 64 → 128 |
| Actor depth limit | ≤ 512 (loss explodes at 1024) |
| Critic depth | Up to 1024 |

```python
config.network.torso = "DeepResidualTorso"
config.network.width = 256
config.network.depth = 64
config.network.activation = "swish"
config.network.dtype = jnp.bfloat16
```

### 4. Scale batch size with depth

| Depth | Batch size |
|-------|-----------|
| 4–8   | 256       |
| 16–32 | 512       |
| 64+   | 1024–2048 |

### 5. Verify stability

- **Actor loss exploding**: reduce actor depth, keep critic deeper
- **Training instability**: confirm LayerNorm on, activation is `"swish"`
- **No improvement from depth**: check residual connections are active
