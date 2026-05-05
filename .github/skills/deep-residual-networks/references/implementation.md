# Deep Residual Network — Implementation Details

Source: [wang-kevin3290/scaling-crl/blob/main/train.py](https://github.com/wang-kevin3290/scaling-crl/blob/main/train.py)

## Full `DeepResidualTorso` implementation

Drop this into `src/coop_rl/networks/torso.py`:

```python
import chex
from flax import linen as nn
from flax.linen.initializers import variance_scaling

from coop_rl.networks.utils import parse_activation_fn

# LeCun uniform: variance_scaling(1/3, fan_in, uniform)
# Paper uses this for all Dense layers (including inside residual blocks)
_lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")


def _residual_block(
    x: chex.Array,
    width: int,
    normalize,
    activation,
) -> chex.Array:
    """One residual block = 4 Dense layers + additive skip.

    Paper: each block has 4 (Dense → LayerNorm → Swish) units plus x + identity.
    Bias is absorbed by LayerNorm so use_bias=False when normalizing.
    """
    identity = x
    for _ in range(4):
        x = nn.Dense(width, kernel_init=_lecun_uniform, use_bias=False)(x)
        x = normalize(x)
        x = activation(x)
    return x + identity


class DeepResidualTorso(nn.Module):
    """Deep residual MLP torso from Wang et al. (NeurIPS 2025 Best Paper).

    Scales to 1000+ layers by combining:
      - Residual connections (additive skip after every 4 Dense layers)
      - LayerNorm after every Dense layer
      - Swish activation (Ramachandran et al. 2018)
      - LeCun uniform initialization

    Args:
        width:          Hidden size of every Dense layer. 256 is parameter-efficient;
                        width 4096 (depth 4) is outperformed by width 256 (depth 8).
        depth:          Total number of Dense layers, must be a multiple of 4.
                        Depth 16–64 is a good starting range for most RL tasks.
                        Critic can go to 1024; keep actor ≤ 512.
        activation:     Activation function name. 'swish' strongly preferred; 'relu'
                        degrades scaling (confirmed by ablation in the paper).
        use_layer_norm: Must be True for stable deep training. Setting False collapses
                        performance at depth ≥ 16 (confirmed by ablation).
    """

    width: int = 256
    depth: int = 16
    activation: str = "swish"
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        assert self.depth % 4 == 0, f"depth must be a multiple of 4, got {self.depth}"

        normalize = nn.LayerNorm() if self.use_layer_norm else (lambda v: v)
        act = parse_activation_fn(self.activation)

        # Input projection
        x = nn.Dense(self.width, kernel_init=_lecun_uniform, use_bias=False)(x)
        x = normalize(x)
        x = act(x)

        # Residual blocks (depth // 4 blocks × 4 layers each = depth total layers)
        for _ in range(self.depth // 4):
            x = _residual_block(x, self.width, normalize, act)

        return x
```

## Parameter count reference

| depth | width | parameters (approx) |
|-------|-------|---------------------|
| 4     | 256   | ~0.5 M              |
| 16    | 256   | ~2 M                |
| 64    | 256   | ~8 M                |
| 256   | 256   | ~33 M               |
| 64    | 1024  | ~134 M              |

Depth 64 width 256 (~8 M) matches or exceeds depth 4 width 4096 (~134 M) in performance.

## Optional: Hyperspherical normalization (SimBa-v2)

SimBa-v2 replaces LayerNorm with hyperspherical normalization — projecting weights onto the unit-norm hypersphere after each gradient update. The paper shows this further improves **sample efficiency** (reaches threshold performance ~15% faster) while retaining the same depth-scaling trend.

To integrate, replace `nn.LayerNorm()` with a custom normalization that applies weight normalization at each step. This is not yet implemented in coop-rl; LayerNorm is the robust default.

## Ablation results (from paper)

| Component removed | Effect |
|------------------|--------|
| LayerNorm         | Severe degradation at all depths ≥ 16 |
| Swish → ReLU      | Significant degradation, especially at depth ≥ 32 |
| Residual connections | Training diverges at depth ≥ 16 |

All three are jointly necessary. Removing any single one is enough to prevent scaling.

## Original residual_block (scaling-crl reference)

This is the exact function from the supplemental code:

```python
lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros

def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x
```

Note: The original uses a non-zero `bias_init` together with LayerNorm (bias is redundant but harmless). The coop-rl adaptation uses `use_bias=False` when LayerNorm is enabled, which is cleaner.

## Training command (scaling-crl reference)

```bash
# depth 16 actor and critic on humanoid
uv run train.py \
  --env_id "humanoid" \
  --critic_depth 16 --actor_depth 16 \
  --actor_skip_connections 4 --critic_skip_connections 4 \
  --batch_size 512
```

`skip_connections` in the original code is the *frequency* of skip connections (every N layers). In our `DeepResidualTorso` the skip is always every 4 layers (one per block), which matches the paper's default.

## Extreme depth notes (from paper §4.3)

- At **1024 layers**: actor loss explodes at training onset. Fix: keep `actor_depth=512`, use `critic_depth=1024` only for the two critic encoders.
- Residual activation norms decrease in deeper layers (known phenomenon from Chang et al. 2018), more pronounced in critics than actors.
- Performance often **jumps discontinuously** at critical depths rather than scaling smoothly — so test at 8 → 16 → 32 → 64 → 128 rather than interpolating.
