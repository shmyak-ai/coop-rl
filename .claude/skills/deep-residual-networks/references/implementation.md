# Deep Residual Network — Implementation Details

Source: [wang-kevin3290/scaling-crl/blob/main/train.py](https://github.com/wang-kevin3290/scaling-crl/blob/main/train.py)

## Full `DeepResidualTorso` implementation

Drop this into `src/coop_rl/networks/torso.py`:

```python
import chex
from flax import linen as nn
from flax.linen.initializers import variance_scaling

from coop_rl.networks.utils import parse_activation_fn

_lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")


def _residual_block(
    x: chex.Array,
    width: int,
    normalize,
    activation,
    dtype: Any,
) -> chex.Array:
    identity = x
    for _ in range(4):
        x = nn.Dense(width, kernel_init=_lecun_uniform, use_bias=False, dtype=dtype)(x)
        x = normalize(x)
        x = activation(x)
    return x + identity


class DeepResidualTorso(nn.Module):
    """Deep residual MLP torso from Wang et al. (NeurIPS 2025 Best Paper).

    Scales to 1000+ layers via residual connections + LayerNorm + Swish + LeCun init.

    Args:
        width:      Hidden size of every Dense layer.
        depth:      Total Dense layers, must be a multiple of 4.
        activation: Activation function name. 'swish' strongly preferred.
        dtype:      Dtype for Dense layers (e.g. jnp.bfloat16).
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

SimBa-v2 replaces LayerNorm with hyperspherical normalization — projecting weights onto the unit-norm hypersphere after each gradient update. ~15% faster to threshold performance. Not yet implemented in coop-rl; LayerNorm is the robust default.

## Ablation results

| Component removed | Effect |
|------------------|--------|
| LayerNorm | Severe degradation at depth ≥ 16 |
| Swish → ReLU | Significant degradation at depth ≥ 32 |
| Residual connections | Training diverges at depth ≥ 16 |

## Extreme depth notes (paper §4.3)

- **1024 layers**: actor loss explodes at onset. Fix: `actor_depth=512`, `critic_depth=1024`.
- Performance often **jumps discontinuously** at critical depths — test 8 → 16 → 32 → 64 → 128.
- Residual activation norms decrease in deeper layers (more pronounced in critics).
