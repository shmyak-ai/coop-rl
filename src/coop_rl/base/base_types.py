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


@chex.dataclass(frozen=True)
class TimeStepDQNRecurrent:
    """TimeStepDQN plus the recurrent state needed for R2D2-style burn-in.

    hidden_state: the RNN carry as it was BEFORE this transition's action was
    selected, so training can warm it back up from the start of a sampled window.
    Only supports single-array carries (e.g. GRU); LSTM's (c, h) tuple carry is
    not representable in this schema.
    reset_hidden_state: done|truncated from the PREVIOUS env step, i.e. whether
    the hidden state should be reset before consuming this transition's obs.
    Not derivable from this window's own terminated/truncated, since it refers
    to the step immediately before the window starts.
    prev_action/prev_reward: action and reward from the env step immediately
    before this transition (zero at episode start), fed into the RNN R2D2-style.
    Stored per transition for the same reason as reset_hidden_state: for the
    first step of a sampled window they are not derivable from the window.
    """

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    terminated: chex.Array
    truncated: chex.Array
    hidden_state: chex.Array
    reset_hidden_state: chex.Array
    prev_action: chex.Array
    prev_reward: chex.Array


class TimeStepDQNRecurrentDtypesAtari(NamedTuple):
    obs: str = "uint8"
    action: str = "int8"
    reward: str = "int32"
    terminated: str = "int8"
    truncated: str = "int8"
    hidden_state: str = "float32"
    reset_hidden_state: str = "int8"
    prev_action: str = "int8"
    prev_reward: str = "int32"


class Observation(NamedTuple):
    """The observation that the agent sees.
    agent_view: the agent's view of the environment.
    action_mask: boolean array specifying which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_view: chex.Array  # (num_obs_features,)
    action_mask: chex.Array  # (num_actions,)
    step_count: chex.Array | None = None  # (,)


# (obs, prev_action, prev_reward, done)
RNNObservation: TypeAlias = tuple[Observation, Action, chex.Array, Done]

ActorApply = Callable[..., Any]
