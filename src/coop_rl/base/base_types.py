from collections.abc import Callable
from typing import Any, TypeAlias

import chex
from typing_extensions import NamedTuple

Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
Truncated: TypeAlias = chex.Array


@chex.dataclass(frozen=True)
class TimeStepDQN:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    terminated: chex.Array
    truncated: chex.Array


class TimeStepDQNDtypesAtari(NamedTuple):
    obs: str = "uint8"
    action: str = "int8"
    reward: str = "int32"
    terminated: str = "int8"
    truncated: str = "int8"


class Observation(NamedTuple):
    """The observation that the agent sees.
    agent_view: the agent's view of the environment.
    action_mask: boolean array specifying which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_view: chex.Array  # (num_obs_features,)
    action_mask: chex.Array  # (num_actions,)
    step_count: chex.Array | None = None  # (,)


RNNObservation: TypeAlias = tuple[Observation, Done]

ActorApply = Callable[..., Any]
