# coop-rl

Distributed cooperative RL framework: Ray (distributed), JAX (computation), Flax (networks).

## Supplements to Global Rules

These extend `~/.claude/CLAUDE.md` — do not repeat the global rules, only apply these additions:

**Simplicity**: Do not implement behavior that was not explicitly requested. Do not replace or remove an existing subsystem unless that exact replacement/removal was explicitly requested. If an additional change seems necessary, ask first. If requirements are ambiguous, ask a clarifying question before coding.

**Surgical changes**: Every changed line must trace directly to the user's request.

## Python Style

Pure functions for training logic, loss computation, and data transforms. Explicit data flow — pass inputs and return outputs. Isolate side effects (I/O, logging, env interaction, device placement, RNG seeding) at system boundaries. OOP is appropriate and encouraged for Ray actors, Flax modules, and stateful lifecycle management. Do not force functional rewrites of well-structured class-based code.

## Protected Files & Quality

- Never modify `uv.lock`, `requirements*.txt`, `*.egg-info/**` unless explicitly asked. If a change seems to require it, stop and ask.
- After every edit: `ruff check --select I --fix && ruff format <changed files>`
- Before reporting done: `pyright <changed files>`, fix all type errors.

## Conventions

### Core Principles

1. **Functional-first**: Pure functions and closures for algorithms. OOP only for stateful Ray actors or Flax modules.
2. **Type-driven**: Explicit TypeAliases, dense annotations, NamedTuples for all data bundles.
3. **JAX immutability**: NamedTuples + `.replace()`, FrozenDict params, pytree-compatible structures.
4. **Decoupled**: Algorithms independent from distributed infrastructure. Pass pure functions into Ray actors.
5. **Configuration-centric**: `ml_collections.ConfigDict` for all experiments.

### Import Order (enforced by ruff)

1. Standard library (`typing`, `logging`, etc.)
2. JAX ecosystem (`jax`, `jax.numpy`, `flax`, `optax`)
3. External scientific (`chex`, `numpy`, `ml_collections`)
4. Project (`coop_rl.*`)

### Type System

```python
Action: TypeAlias = chex.Array

class TrainState(NamedTuple):
    params: FrozenDict
    target_params: FrozenDict
    opt_state: Any
    step: int
```

Always annotate function parameters and returns, especially closures.

### Naming

| Pattern | Usage |
|---------|-------|
| `get_*()` | Factory functions returning closures or objects |
| `_*` | Module-private functions |
| `snake_case` | Functions, variables |
| `PascalCase` | Classes, NamedTuples |
| `CONST_CASE` | Constants |

### Functional Patterns

Canonical update loop:
```python
def get_update_step(config: ConfigDict, apply_fn) -> Callable:
    def _update_step(train_state: TrainState, batch: dict) -> tuple[TrainState, dict]:
        def _loss_fn(params: FrozenDict) -> tuple[chex.Array, dict]:
            pred = apply_fn({"params": params}, batch["observations"])
            loss = jnp.mean((pred - batch["targets"]) ** 2)
            return loss, {"loss": loss}

        grads, info = jax.grad(_loss_fn, has_aux=True)(train_state.params)
        updates, new_opt_state = optimizer.update(grads, train_state.opt_state)
        new_params = optax.apply_updates(train_state.params, updates)
        target_params = optax.incremental_update(
            train_state.params, train_state.target_params, train_state.tau
        )
        return train_state.replace(
            params=new_params, target_params=target_params,
            opt_state=new_opt_state, step=train_state.step + 1,
        ), info
    return _update_step
```

### JAX Patterns

- `@jax.jit` on update functions
- `jax.grad(..., has_aux=True)` for loss + metrics
- Always split PRNG: `rng, subkey = jax.random.split(rng)`
- `jax.tree.map()` for pytree operations
- Prefer `einops` and `jnp.einsum` over manual reshape/indexing:
  ```python
  x = rearrange(x, "b t h w c -> b (t h w) c")   # ✓
  x = x.reshape(x.shape[0], -1, x.shape[-1])       # ✗
  ```

### Flax Modules

Extend `nn.Module`. Use `@nn.compact` or `setup()`. Always type-annotate `__call__`.

### Configuration

```python
def get_config() -> ConfigDict:
    config = ConfigDict()
    config.agent.learning_rate = 1e-4
    config.agent.gamma = 0.99
    config.buffer.size = int(1e6)
    return config
```

### Ray Actors

```python
@ray.remote
class DQNCollector:
    def collect(self, params: FrozenDict) -> dict:
        select_action = get_select_action_fn(self.config)  # pure fn passed in
        return self.env.rollout(select_action, params, self.config.steps)
```

Return pytree-compatible structures (dicts, NamedTuples).

### Antipatterns

| Avoid | Do Instead |
|-------|-----------|
| Mutable state in functions | NamedTuple + `.replace()` |
| `jax.numpy as np` | `import jax.numpy as jnp` |
| Algorithms inside Ray workers | Pass pure functions into actors |
| Bare dicts for state | NamedTuples (type-safe, pytree-compatible) |
| Manual gradient accumulation | `jax.grad(..., has_aux=True)` |
| Mutating parameters | FrozenDict + `.replace()` |

### Testing & Validation

- `chex.assert_shape()` for shape debugging
- `jax.tree_util.tree_all()` for validation
- Test with small arrays to catch shape mismatches early

## Skills

### Deep Residual Networks

@.claude/skills/deep-residual-networks/SKILL.md
