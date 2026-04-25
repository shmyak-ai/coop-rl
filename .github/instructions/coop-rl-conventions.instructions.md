---
name: coop-rl-conventions
description: Code conventions and patterns for coop-rl distributed RL framework. Use when: writing agents, networks, workers, or loss functions; implementing new RL algorithms; working with Ray-based distributed components; adding type definitions or configuration. Apply to Python files in coop_rl/ and app/.
applyTo: "**/coop_rl/**/*.py,**/app/**/*.py"
---

# coop-rl Coding Conventions

This project is a distributed cooperative reinforcement learning framework built on **Ray**, **JAX**, and **Flax**. The codebase favors **functional-first design** with strict type discipline and pure function composition.

## Core Principles

1. **Functional-first approach**: Pure functions, closures for behavioral patterns. OOP only for stateful Ray actors or Flax network modules.
2. **Type-driven clarity**: Explicit TypeAliases for domain concepts, dense type annotations.
3. **JAX immutability**: NamedTuples, FrozenDict parameters, pytree-compatible data structures.
4. **Decoupled architecture**: Algorithms independent from distributed infrastructure.
5. **Configuration-centric**: ml_collections.ConfigDict for reproducible experiments.

---

## 1. Import Organization

**Order** (enforced by ruff):
1. Standard library (typing, logging, collections, etc.)
2. JAX ecosystem (jax, jax.numpy, flax.linen, optax)
3. External scientific (chex, numpy, ml_collections)
4. Project imports (coop_rl modules)

**Examples**:
```python
from typing import Any, Callable, NamedTuple, TypeAlias
import logging

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import chex
from ml_collections import ConfigDict

from coop_rl.base_types import Action, Value, State, TimeStep
from coop_rl.agents import DQNAgent
```

---

## 2. Type System

### Type Aliases
Use TypeAliases for domain concepts to improve readability:

```python
from typing import TypeAlias

# In base_types.py or module scope
Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
State: TypeAlias = Any
ActFn: TypeAlias = Callable[[FrozenDict, Observation, chex.PRNGKey], chex.Array]
```

### NamedTuples for Immutable State
Always use NamedTuples for data bundles (pytree-compatible):

```python
from typing import NamedTuple

class TimeStep(NamedTuple):
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    discount: chex.Array
    done: chex.Array

class TrainState(NamedTuple):
    params: FrozenDict
    target_params: FrozenDict  # For target networks
    opt_state: Any
    step: int
```

### Function Signatures
Always annotate parameters and returns, especially for closures:

```python
def get_loss_fn(config: ConfigDict) -> Callable[[FrozenDict, chex.Array], tuple[chex.Array, dict]]:
    """Returns a loss function."""
    def _loss_fn(params: FrozenDict, batch: chex.Array) -> tuple[chex.Array, dict]:
        # Loss computation
        return loss, {"loss_mean": loss.mean()}
    return _loss_fn
```

---

## 3. Naming Conventions

| Pattern | Usage | Example |
|---------|-------|---------|
| `get_*()` | Factory functions returning closures or objects | `get_update_step()`, `get_actor()`, `get_config()` |
| `_*` | Private functions (module-local) | `_q_loss_fn()`, `_compute_td_error()` |
| `snake_case` | Functions, variables | `update_step`, `target_params`, `buffer_size` |
| `PascalCase` | Classes, NamedTuples | `TrainState`, `DQNAgent`, `FeedForwardActor` |
| `CONST_CASE` | Constants | `DEFAULT_GAMMA = 0.99`, `MAX_STEPS = 1e7` |

---

## 4. Functional Patterns

### Update Step Pattern
The canonical RL update loop uses nested functions with JAX transformations:

```python
def get_update_step(config: ConfigDict, apply_fn) -> Callable:
    """Factory returning an update function."""
    def _update_step(
        train_state: TrainState,
        batch: dict,
    ) -> tuple[TrainState, dict]:
        """Single gradient step."""
        def _loss_fn(params: FrozenDict) -> tuple[chex.Array, dict]:
            # Compute predictions and targets
            pred = apply_fn({"params": params}, batch["observations"])
            loss = jnp.mean((pred - batch["targets"]) ** 2)
            return loss, {"loss": loss}

        # Compute gradients
        grads, info = jax.grad(_loss_fn, has_aux=True)(train_state.params)
        
        # Soft update target network
        target_params = optax.incremental_update(
            train_state.params, train_state.target_params, train_state.tau
        )
        
        # Apply optimizer
        updates, new_opt_state = optimizer.update(grads, train_state.opt_state)
        new_params = optax.apply_updates(train_state.params, updates)
        
        new_train_state = train_state.replace(
            params=new_params,
            target_params=target_params,
            opt_state=new_opt_state,
            step=train_state.step + 1,
        )
        return new_train_state, info
    
    return _update_step
```

### Closures for Behavioral Configuration
Use closures to embed algorithm-specific logic:

```python
def get_select_action_fn(config: ConfigDict) -> Callable[[chex.Array, chex.PRNGKey], chex.Array]:
    """Returns an action selection function parameterized by config."""
    epsilon = config.epsilon
    
    def _select_action(q_values: chex.Array, rng: chex.PRNGKey) -> chex.Array:
        _, rng = jax.random.split(rng)
        return jnp.where(
            jax.random.uniform(rng) < epsilon,
            jax.random.randint(rng, (), 0, q_values.shape[-1]),
            jnp.argmax(q_values),
        )
    
    return _select_action
```

---

## 5. JAX-Specific Patterns

### JIT Compilation
Mark update functions with `@jax.jit` for performance:

```python
@jax.jit
def _update_step(train_state: TrainState, batch: dict) -> tuple[TrainState, dict]:
    # Update logic here
    pass
```

### Gradient Computation
Use `jax.grad(..., has_aux=True)` to return loss + metrics:

```python
grads, (loss, metrics) = jax.grad(_loss_fn, has_aux=True)(params)
```

### RNG Management
Always split the PRNG key:

```python
rng, subkey = jax.random.split(rng)
action = policy_apply(params, obs, key=subkey)
```

### PyTree Operations
Use `jax.tree.map()` for applying functions across nested structures:

```python
def reset_optimizer_state(train_state: TrainState) -> TrainState:
    zero_opt_state = jax.tree.map(jnp.zeros_like, train_state.opt_state)
    return train_state.replace(opt_state=zero_opt_state)
```

---

## 6. Flax Network Modules

Always extend `flax.linen.Module`:

```python
from flax import linen as nn

class FeedForwardActor(nn.Module):
    """Actor network with configurable layers."""
    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = nn.relu
    
    def setup(self):
        """Lazy initialization."""
        self.layers = [
            nn.Dense(dim, name=f"dense_{i}")
            for i, dim in enumerate(self.hidden_dims)
        ]
        self.output_layer = nn.Dense(self.output_dim, name="output")
    
    def __call__(self, x: chex.Array) -> chex.Array:
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.output_layer(x)
```

---

## 7. Configuration Management

Use `ml_collections.ConfigDict` for experiment definitions:

```python
from ml_collections import ConfigDict

def get_config() -> ConfigDict:
    config = ConfigDict()
    config.agent = ConfigDict()
    config.agent.learning_rate = 1e-4
    config.agent.gamma = 0.99
    config.agent.tau = 0.995  # Target network update rate
    
    config.buffer = ConfigDict()
    config.buffer.size = int(1e6)
    config.buffer.batch_size = 32
    
    return config
```

---

## 8. Distributed Components (Ray)

Ray actors use OOP for state management:

```python
import ray

@ray.remote
class DQNCollector:
    def __init__(self, config: ConfigDict):
        self.config = config
        self.env = Environment(config.env_name)
    
    def collect(self, params: FrozenDict) -> dict:
        """Collect trajectories from environment."""
        # Pure function for acting passed into loop
        select_action = get_select_action_fn(self.config)
        
        # Collection logic
        trajectories = self.env.rollout(select_action, params, self.config.steps)
        return trajectories
```

**Key patterns**:
- Use type hints for serialization clarity
- Pass pure functions into remote actors
- Return pytree-compatible structures (dicts, NamedTuples)

---

## 9. Data Structures

### Immutability
Always use immutable structures; use `.replace()` for updates:

```python
# ✓ Correct: NamedTuple update
new_state = train_state.replace(params=new_params, step=train_state.step + 1)

# ✗ Avoid: Mutation
train_state.params = new_params
```

### Ring Buffer Data
Use Flashbax for trajectories:

```python
from coop_rl.buffers import BufferTrajectory

trajectory = BufferTrajectory(
    observations=obs_batch,
    actions=action_batch,
    rewards=reward_batch,
    dones=done_batch,
)
```

---

## 10. Testing & Validation

### Type Checking
Rely on type annotations; ensure files pass Pylance/pyright checks.

### JAX Patterns
- Use `chex.assert_shape()` for shape debugging
- Use `jax.tree_util.tree_all()` for validation
- Test with small arrays to catch shape mismatches early

---

## Antipatterns to Avoid

| ✗ Avoid | ✓ Do Instead | Reason |
|---------|--------------|--------|
| Mutable state in functions | Use NamedTuple + `.replace()` | JAX requires immutability for jit/grad |
| `jax.numpy` as `np` | Import as `import jax.numpy as jnp` | Clarity; distinguishes from NumPy |
| Algorithms in workers | Pass pure functions to Ray actors | Decouples logic from infrastructure |
| Bare dicts for state | Use NamedTuples | Type safety; pytree-compatible |
| Manual gradient accumulation | Use `jax.grad(..., has_aux=True)` | Cleaner, prevents bugs |
| Mutating parameters | Use `FrozenDict` and `.replace()` | JAX requirement; prevents bugs |

---

## Example: Complete DQN Update

```python
from typing import NamedTuple, TypeAlias
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from ml_collections import ConfigDict

# Type definitions
QApply: TypeAlias = Callable[[FrozenDict, chex.Array], chex.Array]

class DQNTrainState(NamedTuple):
    params: FrozenDict
    target_params: FrozenDict
    opt_state: Any
    step: int

def get_dqn_update(config: ConfigDict, q_apply: QApply) -> Callable:
    """Create a DQN update function."""
    def _update(
        train_state: DQNTrainState,
        batch: dict,
        key: chex.PRNGKey,
    ) -> tuple[DQNTrainState, dict]:
        def _loss_fn(params: FrozenDict) -> tuple[chex.Array, dict]:
            # Compute TD error
            q_pred = q_apply(params, batch["obs"])
            q_target = q_apply(train_state.target_params, batch["next_obs"])
            
            td_target = batch["reward"] + config.gamma * jnp.max(q_target, axis=-1) * (1 - batch["done"])
            td_error = q_pred - td_target[(..., None)]
            loss = jnp.mean(td_error ** 2)
            
            return loss, {"loss": loss, "td_error": jnp.mean(jnp.abs(td_error))}
        
        grads, info = jax.grad(_loss_fn, has_aux=True)(train_state.params)
        updates, opt_state = optimizer.update(grads, train_state.opt_state)
        params = optax.apply_updates(train_state.params, updates)
        
        target_params = optax.incremental_update(
            params, train_state.target_params, config.tau
        )
        
        new_state = train_state.replace(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            step=train_state.step + 1,
        )
        return new_state, info
    
    return _update
```

---

## See Also
- [base_types.py](coop_rl/base_types.py) – Core type definitions
- [agents/dqn.py](coop_rl/agents/dqn.py) – Reference DQN implementation
- [agents/dreamer.py](coop_rl/agents/dreamer.py) – Complex world model implementation
- [workers/collectors.py](coop_rl/workers/collectors.py) – Distributed collection pattern
