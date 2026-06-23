from functools import partial

import elements
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


class FirstDimToLast(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_obs_space_shape = list(env.observation_space.shape[1:])
        new_obs_space_shape.append(env.observation_space.shape[0])
        orig = env.observation_space
        self.observation_space = Box(
            low=np.moveaxis(orig.low, 0, -1),
            high=np.moveaxis(orig.high, 0, -1),
            dtype=orig.dtype,
        )

    def observation(self, obs):
        transposed_obs = np.moveaxis(obs, 0, -1)
        return transposed_obs


class HandlerEnv:
    def __init__(self, env_name, stack_size):
        self._env = HandlerEnv.make_env(env_name, stack_size)

    def reset(self, *args, seed=None, **kwargs):
        observation, info = self._env.reset(*args, seed=seed, **kwargs)
        # call [:] to get a stacked array from gym
        return observation[:], info

    def step(self, action, *args, **kwargs):
        observation, reward, terminated, truncated, info = self._env.step(action)
        return observation[:], reward, terminated, truncated, info

    @staticmethod
    def make_env(env_name, stack_size):
        env = gym.make(env_name)
        if stack_size > 1:
            env = FrameStackObservation(env, stack_size)
        return env

    @staticmethod
    def check_env(env_name, stack_size, *args, **kwargs):
        env = HandlerEnv.make_env(env_name, stack_size)
        return (
            env.observation_space.shape,
            env.observation_space.dtype,
            env.action_space.n,
        )


def _make_atari_env(
    env_name,
    stack_size,
    kwargs,
    *,
    screen_size=84,
    grayscale=True,
    add_channel=False,
    repeat_action_probability=0.0,
):
    """Module-level factory for VectorEnv (must be picklable)."""
    import ale_py

    ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    env = gym.make(
        env_name, frameskip=1, repeat_action_probability=repeat_action_probability, **kwargs
    )
    env = AtariPreprocessing(
        env,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=grayscale,
        grayscale_newaxis=add_channel,
        scale_obs=False,
    )
    if stack_size > 1:
        env = FrameStackObservation(env, stack_size)
        env = FirstDimToLast(env)
    return env


class HandlerEnvAtari:
    """Atari env handler supporting 1 or more parallel environments.

    When ``num_envs=1`` a ``SyncVectorEnv`` is used (no subprocess overhead).
    When ``num_envs>1`` an ``AsyncVectorEnv`` is used (parallel subprocesses).
    ``reset`` and ``step`` always return batched arrays of shape
    ``(num_envs, *obs_shape)``.
    """

    def __init__(self, env_name, *, stack_size=1, num_envs=1, **kwargs):
        factory = partial(_make_atari_env, env_name, stack_size, kwargs)
        # DISABLED mode: env.step() never auto-resets sub-environments.
        # The collector resets done envs explicitly, preventing cross-episode
        # transitions from being stored in the replay buffer (which happens
        # with the default NextStep mode).
        autoreset_mode = gym.vector.AutoresetMode.DISABLED
        if num_envs > 1:
            self._env = gym.vector.AsyncVectorEnv(
                [factory] * num_envs, context="forkserver", autoreset_mode=autoreset_mode
            )
        else:
            self._env = gym.vector.SyncVectorEnv([factory], autoreset_mode=autoreset_mode)
        self.num_envs = num_envs

    def reset(self, *, seed=None, **kwargs):
        obs, info = self._env.reset(seed=seed, **kwargs)
        return obs, info  # (num_envs, *obs_shape)

    def step(self, actions):  # actions: (num_envs,)
        obs, rewards, terminated, truncated, infos = self._env.step(actions)
        return obs, rewards, terminated, truncated, infos

    @staticmethod
    def make_env(env_name, stack_size, **kwargs):
        return _make_atari_env(env_name, stack_size, kwargs)

    @staticmethod
    def check_env(env_name, stack_size, num_envs=1, **kwargs):
        env = _make_atari_env(env_name, stack_size, kwargs)
        shape = env.observation_space.shape
        dtype = env.observation_space.dtype
        n_actions = env.action_space.n
        env.close()
        return shape, dtype, n_actions

    def close(self):
        self._env.close()


class HandlerEnvDreamerAtari:
    """Gymnasium-based Atari env handler for DreamerV3.

    Single grayscale frame ``(screen_size, screen_size, 1)`` uint8, NEXT_STEP
    autoreset (the done step returns the terminal obs, the next step returns the
    fresh reset obs) so the collector can mark ``is_first`` from the previous
    step's done flag. ``check_env`` returns the dict-of-Spaces obs/act contract
    the Dreamer buffer and world model expect.
    """

    def __init__(self, *, env_name, num_envs=1, screen_size=96, sticky=True):
        rap = 0.25 if sticky else 0.0
        factory = partial(
            _make_atari_env,
            env_name,
            1,
            {},
            screen_size=screen_size,
            grayscale=True,
            add_channel=True,
            repeat_action_probability=rap,
        )
        autoreset_mode = gym.vector.AutoresetMode.NEXT_STEP
        if num_envs > 1:
            self._env = gym.vector.AsyncVectorEnv(
                [factory] * num_envs, context="forkserver", autoreset_mode=autoreset_mode
            )
        else:
            self._env = gym.vector.SyncVectorEnv([factory], autoreset_mode=autoreset_mode)
        self.num_envs = num_envs

    def reset(self, *, seed=None, **kwargs):
        obs, info = self._env.reset(seed=seed, **kwargs)
        return obs, info  # (num_envs, screen_size, screen_size, 1)

    def step(self, actions):  # actions: (num_envs,)
        return self._env.step(actions)  # obs, rewards, terminated, truncated, infos

    @staticmethod
    def make_env(env_name, screen_size=96, sticky=True):
        rap = 0.25 if sticky else 0.0
        return _make_atari_env(
            env_name,
            1,
            {},
            screen_size=screen_size,
            grayscale=True,
            add_channel=True,
            repeat_action_probability=rap,
        )

    def close(self):
        self._env.close()

    @staticmethod
    def check_env(*, env_name, num_envs=1, screen_size=96, sticky=True):
        env = HandlerEnvDreamerAtari.make_env(env_name, screen_size=screen_size, sticky=sticky)
        shape = env.observation_space.shape
        n_actions = env.action_space.n
        env.close()
        obs_space = {
            "image": elements.Space(np.uint8, shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        act_space = {"action": elements.Space(np.int32, (), 0, int(n_actions))}
        return obs_space, None, act_space


def _make_godot_env(env_path, port, show_window, speedup, seed, num_envs):
    """Connect to a Godot godot_rl environment hosting ``num_envs`` agents.

    Imports ``godot_rl`` lazily so this module stays importable without the
    optional dependency (mirrors ``_make_atari_env`` / ``_make_unity_env``).
    Python is the TCP *server*: with ``env_path=None`` this blocks on ``accept()``
    until you press Play in the Godot editor; with a built binary path GodotEnv
    launches the game itself. ``n_envs`` is forwarded as ``--n_envs=N`` to the
    binary (ignored in editor mode, where the scene's own count applies);
    ``convert_action_space=False`` keeps the action as the dict the game reports
    (e.g. ``{'act': Discrete(6)}``).
    """
    from godot_rl.core.godot_env import GodotEnv

    return GodotEnv(
        env_path=env_path,
        port=port,
        show_window=show_window,
        speedup=speedup,
        seed=seed,
        n_envs=num_envs,
        convert_action_space=False,
    )


def _single_image_key(observation_space) -> str:
    keys = [
        k
        for k, s in observation_space.spaces.items()
        if isinstance(s, Box) and len(s.shape) == 3 and s.dtype == np.uint8
    ]
    if len(keys) != 1:
        raise ValueError(
            f"HandlerGodotEnv requires exactly one 3-D uint8 image observation; found {keys}."
        )
    return keys[0]


def _single_discrete_n(action_space) -> int:
    heads = list(action_space.spaces.items())
    if len(heads) != 1 or not isinstance(heads[0][1], Discrete):
        raise ValueError(
            f"HandlerGodotEnv supports a single discrete action head; got {action_space}."
        )
    return int(heads[0][1].n)


def _godot_image(obs_chw, pad_to):
    """Channel-first uint8 ``(C, H, W)`` -> channels-last ``(pad_to, pad_to, C)``.

    The Godot map is channel-major (e.g. ``map_2d`` (6, 25, 25)); Dreamer's conv
    encoder wants channels-last and a resolution that is even and divisible by 16,
    so the grid is placed top-left and zero-padded (0 == empty per the env schema).
    """
    img = np.transpose(obs_chw, (1, 2, 0))  # (H, W, C)
    h, w = img.shape[:2]
    if h > pad_to or w > pad_to:
        raise ValueError(f"image {img.shape} larger than pad_to={pad_to}.")
    return np.pad(img, ((0, pad_to - h), (0, pad_to - w), (0, 0))).astype(np.uint8)


class HandlerGodotEnv:
    """godot_rl env handler for the DreamerV3 pipeline (vector of N agents).

    Wraps one ``GodotEnv`` connected to a Godot game embedding the
    ``godot_rl_agents`` plugin. A single process hosts ``num_envs`` ``AIController``
    agents (Bridge: ``voxel_terrain.gd::NUM_ENVS`` / ``--n_envs``), so obs, reward
    and done arrive as length-N arrays. Each agent's 3-D uint8 observation
    (channel-first, e.g. ``map_2d`` (6, 25, 25)) is transposed to channels-last and
    zero-padded to ``(pad_to, pad_to, C)`` so it fits the Dreamer conv encoder
    (which needs an even resolution divisible by 16), then stacked to
    ``(N, pad_to, pad_to, C)``. Scale to K×N by running K collectors, each on its
    own port (see ``runtime.training``).

    Python is the TCP *server*: with ``env_path=None`` construction blocks until
    you press Play in the Godot editor (single process only); with a built binary
    path GodotEnv launches the game itself headless.

    Bridge auto-resets each sub-env internally the moment that agent reports
    ``done`` (no Python-side reset), so this handler passes obs/reward/done
    straight through and the collector treats it as a gymnasium auto-resetting
    vector env. godot_rl reports a single ``done`` as both ``terminated`` and
    ``truncated``; both are passed through unchanged. (godot_rl does not preserve a
    distinct terminal observation across the internal reset — a known plugin
    limitation, see the Bridge trainer-wrapper doc.)
    """

    def __init__(
        self,
        *,
        env_path=None,
        num_envs,
        port=11008,
        show_window=False,
        speedup=1,
        seed=0,
        pad_to=48,
        image_channels,
        num_actions,
    ):
        self.pad_to = pad_to
        self._env = _make_godot_env(env_path, port, show_window, speedup, seed, num_envs)
        if self._env.num_envs != num_envs:
            raise ValueError(
                f"Godot scene reports {self._env.num_envs} agents but config num_envs={num_envs}; "
                "they must match (the buffer add_batch_size is sized from num_envs)."
            )
        self.num_envs = num_envs
        self._image_key = _single_image_key(self._env.observation_space)
        live_channels = int(self._env.observation_space[self._image_key].shape[0])
        live_actions = _single_discrete_n(self._env.action_spaces[0])
        if live_channels != image_channels or live_actions != num_actions:
            raise ValueError(
                "Godot game schema does not match config: live image_channels="
                f"{live_channels}, num_actions={live_actions} vs configured "
                f"image_channels={image_channels}, num_actions={num_actions}."
            )

    def _images(self, obs):
        return np.stack([_godot_image(o[self._image_key], self.pad_to) for o in obs])

    def reset(self, *, seed=None, **kwargs):
        obs, _ = self._env.reset(seed=seed)
        return self._images(obs), {}  # (N, pad_to, pad_to, C)

    def step(self, actions):
        acts = [int(a) for a in np.asarray(actions).reshape(-1)]  # one head, N agents
        obs, reward, terminated, truncated, _ = self._env.step([acts])
        return (
            self._images(obs),
            np.asarray(reward, np.float32),
            np.asarray(terminated, bool),
            np.asarray(truncated, bool),
            {},
        )

    def close(self):
        self._env.close()

    @staticmethod
    def check_env(
        *,
        image_channels,
        num_actions,
        pad_to=48,
        env_path=None,
        num_envs=1,
        port=11008,
        show_window=False,
        speedup=1,
        seed=0,
    ):
        # Build the space contract from the fixed Bridge schema (image_channels,
        # num_actions) WITHOUT opening a GodotEnv. A live connection here would
        # consume the single editor Play session (close() quits the game), leaving
        # nothing for the collector's env to connect to. The connection-time params
        # (env_path/port/...) are accepted to share **args_env but unused.
        obs_space = {
            "image": elements.Space(np.uint8, (pad_to, pad_to, image_channels)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        act_space = {"action": elements.Space(np.int32, (), 0, num_actions)}
        return obs_space, None, act_space
