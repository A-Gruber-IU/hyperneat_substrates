from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from typing import Callable, Optional, Tuple, Dict, Any
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import gymnasium as gym
import imageio
import pickle
import csv
import os


# ---------- Minimal State replacement (instead of tensorneat.common.State) ----------
@dataclass(frozen=True)
class RLState:
    """Holds RNG and optional normalization stats."""
    randkey: jax.Array
    problem_obs_mean: Optional[np.ndarray] = None
    problem_obs_std: Optional[np.ndarray] = None

    def register(self, **kwargs) -> "RLState":
        # Functional update (return a new state with provided fields replaced)
        return replace(self, **kwargs)

    def save(self, file_name: str):
        # Move JAX arrays to host for portability
        payload = {
            k: (jax.device_get(v) if isinstance(v, jax.Array) else v)
            for k, v in asdict(self).items()
        }
        with open(file_name, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, file_name: str) -> "RLState":
        with open(file_name, "rb") as f:
            payload = pickle.load(f)
        # Cast arrays back to JAX
        to_jax = lambda x: jnp.asarray(x) if x is not None else None
        return cls(
            randkey=jnp.asarray(payload["randkey"]),
            problem_obs_mean=to_jax(payload["problem_obs_mean"]),
            problem_obs_std=to_jax(payload["problem_obs_std"]),
        )


# ---------- Standalone base RLEnv (no TensorNEAT dependency) ----------
class RLEnv:
    """
    A JAX-friendly RL evaluation harness with optional observation normalization
    and optional action_policy.

    Public API:
      - setup(state: RLState) -> RLState
      - evaluate(state, randkey, act_func, params)
      - env_reset / env_step to be implemented by subclasses
      - input_shape / output_shape properties
      - show(...) optional
    """

    # Keeps parity with the original class attribute (not required by JAX)
    jitable = True

    def __init__(
        self,
        max_step: int = 1000,
        repeat_times: int = 1,
        action_policy: Optional[Callable] = None,
        obs_normalization: bool = False,
        sample_policy: Optional[Callable] = None,
        sample_episodes: int = 0,
    ):
        """
        action_policy(randkey, forward_func, obs) -> action
        forward_func(obs) -> action (wraps act_func(state, params, obs))
        sample_policy(randkey, obs) -> action
        Used only when obs_normalization=True to collect observations.
        """
        self.max_step = max_step
        self.repeat_times = repeat_times
        self.action_policy = action_policy
        self.obs_normalization = obs_normalization
        if obs_normalization:
            assert sample_policy is not None, "sample_policy must be provided when obs_normalization=True"
            assert sample_episodes > 0, "sample_episodes must be > 0 when obs_normalization=True"
        self.sample_policy = sample_policy
        self.sample_episodes = sample_episodes

    # ---- abstract API for subclasses ----
    @property
    def input_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def env_reset(self, randkey: jax.Array):
        raise NotImplementedError

    def env_step(self, randkey: jax.Array, env_state, action):
        raise NotImplementedError

    # ---- generic helpers (mirror original) ----
    def reset(self, randkey):
        return self.env_reset(randkey)

    def step(self, randkey, env_state, action):
        return self.env_step(randkey, env_state, action)

    def setup(self, state: RLState) -> RLState:
        """
        Optionally run sampling rollouts to compute observation mean/std for normalization,
        """
        if not self.obs_normalization:
            return state

        print("Sampling episodes for normalization")
        keys = jax.random.split(state.randkey, self.sample_episodes)

        # Dummy act_func that just returns obs (so we record raw obs)
        dummy_act_func = lambda s, p, o: o
        # sample_policy ignores forward_func; we keep signature compatibility
        dummy_sample_func = lambda rk, forward_func, obs: self.sample_policy(rk, obs)

        def sample_once(rk):
            # record_episode=True to capture obs in episode dict
            return self._evaluate_once(
                state, rk, dummy_act_func, None, dummy_sample_func, True, False
            )

        # Vectorize sampling, jit-compile it
        rewards, episodes = jax.jit(vmap(sample_once))(keys)

        # episodes["obs"] shape: (sample_episodes, max_step, *input_shape)
        obs = jax.device_get(episodes["obs"]).reshape(-1, *self.input_shape)

        obs_axis = tuple(range(obs.ndim))
        valid_mask = np.all(~np.isnan(obs), axis=obs_axis[1:])
        obs = obs[valid_mask]
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0)

        new_state = state.register(problem_obs_mean=obs_mean, problem_obs_std=obs_std)
        print("Sampling episodes for normalization finished.")
        print("valid data count:", obs.shape[0])
        return new_state

    def evaluate(self, state: RLState, randkey, act_func: Callable, params):
        keys = jax.random.split(randkey, self.repeat_times)
        rewards = vmap(
            self._evaluate_once, in_axes=(None, 0, None, None, None, None, None)
        )(state, keys, act_func, params, self.action_policy, False, self.obs_normalization)
        return rewards.mean()

    def _evaluate_once(
        self,
        state: RLState,
        randkey,
        act_func: Callable,
        params,
        action_policy: Optional[Callable],
        record_episode: bool,
        normalize_obs: bool = False,
    ):
        rng_reset, rng_episode = jax.random.split(randkey)
        init_obs, init_env_state = self.reset(rng_reset)

        if record_episode:
            obs_array = jnp.full((self.max_step, *self.input_shape), jnp.nan, dtype=jnp.float32)
            action_array = jnp.full((self.max_step, *self.output_shape), jnp.nan, dtype=jnp.float32)
            reward_array = jnp.full((self.max_step,), jnp.nan, dtype=jnp.float32)
            episode = {"obs": obs_array, "action": action_array, "reward": reward_array}
        else:
            episode = None

        def cond(carry):
            _, _, _, done, _, count, _, _ = carry
            return jnp.logical_and(~done, count < self.max_step)

        def body(carry):
            obs, env_state, rng, done, total_reward, count, epis, rk = carry

            if normalize_obs:
                obs = (obs - state.problem_obs_mean) / (state.problem_obs_std + 1e-6)

            if action_policy is not None:
                forward_func = lambda o: act_func(state, params, o)
                action = action_policy(rk, forward_func, obs)
            else:
                action = act_func(state, params, obs)

            next_obs, next_env_state, reward, done, _ = self.step(rng, env_state, action)
            next_rng, _ = jax.random.split(rng)

            if record_episode:
                epis["obs"] = epis["obs"].at[count].set(obs)
                epis["action"] = epis["action"].at[count].set(action)
                epis["reward"] = epis["reward"].at[count].set(reward)

            return (
                next_obs,
                next_env_state,
                next_rng,
                done,
                total_reward + reward,
                count + 1,
                epis,
                jax.random.split(rk)[0],
            )

        _, _, _, _, total_reward, _, episode, _ = jax.lax.while_loop(
            cond,
            body,
            (init_obs, init_env_state, rng_episode, False, jnp.float32(0.0), 0, episode, randkey),
        )
        return (total_reward, episode) if record_episode else total_reward

    # Subclasses can override if they want a visualization method
    def show(self, state, randkey, act_func, params, *args, **kwargs):
        raise NotImplementedError


class CustomGymnasiumEnv(RLEnv):
    def __init__(
        self,
        env_name: str,
        *,
        # --- RLEnv (training) knobs ---
        max_step: int = 1000,
        repeat_times: int = 1,
        action_policy: Optional[Callable] = None,
        obs_normalization: bool = False,
        sample_policy: Optional[Callable] = None,
        sample_episodes: int = 0,
        # --- Wrapper shaping knobs ---
        control_cost_weight: float = 0.0,
        inaction_cost_weight: float = 0.0,
        alive_bonus: float = 0.0,
        movement_bonus_weight: float = 0.0,
        # --- Gym-specific kwargs go ONLY here ---
        prefer_dense_reward: bool = True,
        env_make_kwargs: Optional[Dict[str, Any]] = None,

        # time/positional features
        time_wavelengths: Optional[Tuple[int, ...]] = None,   # steps per cycle
        time_include_cos: bool = False,
    ):
        # Initialize the base harness first (consumes training-only args)
        super().__init__(
            max_step=max_step,
            repeat_times=repeat_times,
            action_policy=action_policy,
            obs_normalization=obs_normalization,
            sample_policy=sample_policy,
            sample_episodes=sample_episodes,
        )

        self.env_name = env_name
        self.control_cost_weight = float(control_cost_weight)
        self.inaction_cost_weight = float(inaction_cost_weight)
        self.alive_bonus = float(alive_bonus)
        self.movement_bonus_weight = float(movement_bonus_weight)

        # --- create the Gymnasium env (clean kwargs) ---
        env_kwargs = dict(env_make_kwargs or {})
        # Try dense reward first (only if supported). If not supported, retry without it.
        try_kwargs = dict(env_kwargs)
        if prefer_dense_reward and "reward_type" not in try_kwargs:
            try_kwargs["reward_type"] = "dense"
        try:
            self.env = gym.make(env_name, **try_kwargs)
        except TypeError:
            # Remove reward_type and retry — needed for locomotion envs like Ant-v5
            try_kwargs.pop("reward_type", None)
            self.env = gym.make(env_name, **try_kwargs)

        # --- infer base shapes ---
        self._action_size = int(np.prod(self.env.action_space.shape))
        obs_sample, _ = self.env.reset()
        flat_base = self._flatten_obs(obs_sample)
        self._base_obs_shape = flat_base.shape  # keep base shape (important!)

        self._act_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self._act_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        assert self._act_low.shape == self._act_high.shape == self.env.action_space.shape
        self._act_high_jnp = np.asarray(self._act_high, dtype=np.float32)
        self._act_low_jnp = np.asarray(self._act_low, dtype=np.float32)

        # -------- configure time features --------
        self.time_wavelengths = tuple(time_wavelengths or ())
        self.time_include_cos = bool(time_include_cos)

        per_wave = 1 + int(self.time_include_cos)   # sin (+ cos)
        self._time_feat_dim = len(self.time_wavelengths) * per_wave

        # final (augmented) input shape exposed to the algorithm
        self._input_shape = (self._base_obs_shape[0] + self._time_feat_dim,)

        print(
            f"Initialized GymnasiumEnv '{self.env_name}' "
            f"with control_cost={self.control_cost_weight}, inaction_cost={self.inaction_cost_weight}"
        )
        print(f"Action shape: {self.output_shape}")
        print(f"Observation shape (base -> augmented): {self._base_obs_shape} -> {self.input_shape}")

    # ---------- utils ----------
    def _flatten_obs(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            return np.concatenate(
                [obs["observation"], obs["desired_goal"], obs["achieved_goal"]]
            ).astype(np.float32)
        return np.asarray(obs, dtype=np.float32)

    # --- NEW: time feature helpers ---
    def _time_features(self, t: jax.Array) -> jax.Array:
        """Return a 1D float32 vector of time features for timestep t."""
        if self._time_feat_dim == 0:
            return jnp.zeros((0,), dtype=jnp.float32)

        feats = []
        if self.time_wavelengths:
            t32 = t.astype(jnp.float32)
            wl = jnp.asarray(self.time_wavelengths, dtype=jnp.float32)
            angle = 2.0 * jnp.pi * (t32 / wl)      # 2π * t / wavelength
            s = jnp.sin(angle)                     # (K,)
            feats.append(s)
            if self.time_include_cos:
                c = jnp.cos(angle)                 # (K,)
                feats.append(c)

        return jnp.concatenate(feats, axis=0).astype(jnp.float32) if feats else jnp.zeros((0,), dtype=jnp.float32)

    def _augment_obs(self, flat_obs: jax.Array, t: jax.Array) -> jax.Array:
        """Append time features to flat_obs."""
        if self._time_feat_dim == 0:
            return flat_obs
        return jnp.concatenate([flat_obs, self._time_features(t)], axis=0)

    # ---------- RLEnv abstract impls ----------
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self._action_size,)

    def env_reset(self, randkey):
        # NOTE: callback should return BASE obs shape, not augmented.
        result_spec = jax.ShapeDtypeStruct(self._base_obs_shape, jnp.float32)

        def _reset_callback():
            obs, _ = self.env.reset()
            flat_obs = self._flatten_obs(obs)
            return flat_obs

        flat_base = jax.pure_callback(_reset_callback, result_spec, vmap_method="sequential")
        t0 = jnp.array(0, dtype=jnp.int32)
        flat_aug = self._augment_obs(flat_base, t0)

        prev_action = jnp.zeros(self.output_shape, dtype=jnp.float32)
        # env_state packs (prev_action, timestep)
        env_state = (prev_action, t0)
        return flat_aug, env_state

    def env_step(self, randkey, env_state, action):
        prev_action, t = env_state  # unpack state

        # 1) Scale the NN output from [-1, 1] -> [low, high] on the JAX side
        scaled = 0.5 * (action + 1.0) * (self._act_high_jnp - self._act_low_jnp) + self._act_low_jnp
        scaled = jnp.clip(scaled, self._act_low_jnp, self._act_high_jnp)

        # 2) Compute costs on *scaled* values so they reflect actual torques/speeds
        control_cost = self.control_cost_weight * jnp.sum(scaled * scaled)
        inaction_cost = self.inaction_cost_weight * jnp.sum((scaled - prev_action) ** 2)
        total_cost = control_cost + inaction_cost

        # Callback returns BASE obs; we then augment with time features
        result_spec = (
            jax.ShapeDtypeStruct(self._base_obs_shape, jnp.float32),
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )

        def _step_callback(a_np):
            a_np = np.asarray(a_np, dtype=np.float32)
            # (Safety) Clip on the Python side too
            a_np = np.clip(a_np, self._act_low, self._act_high)
            obs, reward, terminated, truncated, _ = self.env.step(a_np)
            done = bool(terminated or truncated)
            shaped = float(reward)
            return (self._flatten_obs(obs), np.float32(shaped), np.bool_(done))

        flat_base, avg_step_shaped_reward, done = jax.pure_callback(
            _step_callback, result_spec, scaled, vmap_method="sequential"
        )

        alive = jnp.where(done, jnp.float32(0.0), jnp.float32(self.alive_bonus))
        movement_bonus = self.movement_bonus_weight * jnp.sum(jnp.abs(scaled))
        avg_step_final_reward = avg_step_shaped_reward + alive - total_cost + movement_bonus

        info = {
            'avg_step_final_reward': avg_step_final_reward,
            'avg_step_shaped_reward': avg_step_shaped_reward, # The original reward from the env
            'alive_bonus': alive,
            'control_cost': control_cost,
            'inaction_cost': inaction_cost,
            'movement_bonus': movement_bonus
        }

        # increment t for the NEXT observation
        t_next = t + jnp.int32(1)
        flat_aug = self._augment_obs(flat_base, t_next)

        next_state = (scaled, t_next)  # store the scaled action and updated t

        return flat_aug, next_state, avg_step_final_reward, done, info

    def show(self, state, act_func, params, **kwargs):
        # Use a separate env instance for rendering
        render_kwargs = dict(kwargs.pop("env_make_kwargs", {}) or {})
        try:
            if "render_mode" not in render_kwargs:
                render_kwargs["render_mode"] = "rgb_array"
            render_env = gym.make(self.env_name, **render_kwargs)
        except TypeError:
            # Fallback without any extra kwargs if necessary
            render_env = gym.make(self.env_name, render_mode="rgb_array")

        save_path = kwargs.get("save_path", f"{self.env_name}.mp4")
        fps = int(kwargs.get("fps", 30))
        frames = []

        obs, _ = render_env.reset()
        total_avg_step_final_reward = 0.0
        prev_action = np.zeros(self.output_shape, dtype=np.float32)
        t = 0  # timestep for time features in show loop

        for _ in range(self.max_step):
            frames.append(render_env.render())

            flat_base = self._flatten_obs(obs)
            flat_aug = np.asarray(self._augment_obs(jnp.asarray(flat_base), jnp.int32(t)))

            if self.obs_normalization:
                if state.problem_obs_mean is not None and state.problem_obs_std is not None:
                    mean = np.asarray(state.problem_obs_mean)
                    std = np.asarray(state.problem_obs_std)
                    flat_aug = (flat_aug - mean) / (std + 1e-6)
                else:
                    print("Warning: obs_normalization is True, but no normalization stats were found in the state. Using raw observations.")

            # policy action (assumed in [-1, 1] like training)
            action_nn = np.asarray(act_func(state, params, jnp.asarray(flat_aug)), dtype=np.float32)

            # scale exactly as in env_step
            action_scaled = 0.5 * (action_nn + 1.0) * (self._act_high - self._act_low) + self._act_low
            action_scaled = np.clip(action_scaled, self._act_low, self._act_high)

            obs, reward, terminated, truncated, _ = render_env.step(action_scaled)

            # mirror reward shaping used in env_step
            control = float(self.control_cost_weight * np.sum(action_scaled * action_scaled))
            inaction = float(self.inaction_cost_weight * np.sum((action_scaled - prev_action) ** 2))
            alive = 0.0 if (terminated or truncated) else float(self.alive_bonus)
            movement_bonus = float(self.movement_bonus_weight * np.sum(np.abs(action_scaled)))
            avg_step_final_reward = float(reward) + alive - (control + inaction) + movement_bonus

            total_avg_step_final_reward += avg_step_final_reward
            prev_action = action_scaled
            t += 1

            if terminated or truncated:
                break

        render_env.close()
        imageio.mimsave(save_path, frames, fps=fps)
        print(f"Video saved to {save_path}")
        print(f"Visualization finished. Total final reward: {total_avg_step_final_reward}")

    def log_trajectory(self, state, randkey, act_func, params, *,
                       save_path: str,
                       log_every_n_steps: int = 1):
        """
        Runs one episode with the given agent and logs observations and actions
        to a CSV file at a specified sampling rate.

        Args:
            state: The RLState.
            randkey: A JAX random key (for API consistency, not used in the loop).
            act_func: The agent's policy function.
            params: The network parameters for the agent.
            save_path: The full path to the output CSV file.
            log_every_n_steps: The sampling rate (e.g., 100 means log every 100th step).
        """
        print(f"Logging trajectory to {save_path} (every {log_every_n_steps} steps)...")

        # Use a separate env instance for logging to avoid state conflicts
        log_env = gym.make(self.env_name)

        # Prepare for logging
        log_records = []
        obs, _ = log_env.reset()
        prev_action = np.zeros(self.output_shape, dtype=np.float32)
        t = 0

        # Dynamically create CSV headers
        header = ['timestep', 'raw_reward', 'final_avg_step_shaped_reward']
        header.extend([f'prev_action_{i}' for i in range(self.output_shape[0])])
        header.extend([f'obs_{i}' for i in range(self.input_shape[0])])
        header.extend([f'action_{i}' for i in range(self.output_shape[0])])

        for t in range(self.max_step):
            flat_base = self._flatten_obs(obs)
            flat_aug = np.asarray(self._augment_obs(jnp.asarray(flat_base), jnp.int32(t)))
            
            action_nn = np.asarray(act_func(state, params, jnp.asarray(flat_aug)), dtype=np.float32)

            action_scaled = 0.5 * (action_nn + 1.0) * (self._act_high - self._act_low) + self._act_low
            action_scaled = np.clip(action_scaled, self._act_low, self._act_high)

            # Log data *before* stepping to capture the state that led to the action
            if t % log_every_n_steps == 0:
                record = {
                    'timestep': t,
                    # We will fill rewards after the step
                }
                record.update({f'prev_action_{i}': val for i, val in enumerate(prev_action)})
                record.update({f'obs_{i}': val for i, val in enumerate(flat_aug)})
                record.update({f'action_{i}': val for i, val in enumerate(action_scaled)})
                log_records.append(record)

            obs, reward, terminated, truncated, _ = log_env.step(action_scaled)
            
            # Calculate final shaped reward to also log it
            control = float(self.control_cost_weight * np.sum(action_scaled * action_scaled))
            inaction = float(self.inaction_cost_weight * np.sum((action_scaled - prev_action) ** 2))
            alive = 0.0 if (terminated or truncated) else float(self.alive_bonus)
            movement_bonus = float(self.movement_bonus_weight * np.sum(np.abs(action_scaled)))
            avg_step_final_reward = float(reward) + alive - (control + inaction) + movement_bonus

            # Update the record with the rewards
            if t % log_every_n_steps == 0:
                # Find the corresponding record (the last one added)
                log_records[-1]['raw_reward'] = float(reward)
                log_records[-1]['final_avg_step_shaped_reward'] = avg_step_final_reward

            prev_action = action_scaled
            
            if terminated or truncated:
                break

        log_env.close()

        # Write to CSV
        if not log_records:
            print("Warning: No data was logged. The episode may have ended before the first sample.")
            return
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(log_records)
            
        print("Trajectory logging complete.")