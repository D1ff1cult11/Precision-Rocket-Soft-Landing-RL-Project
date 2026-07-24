from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.config import EnvConfig


class PrecisionRocketEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        high = np.array([5.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.state: Optional[np.ndarray] = None
        self.prev_potential: float = 0.0
        self.step_count: int = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        y = self.np_random.uniform(3.5, 4.5)
        vy = self.np_random.uniform(-1.0, 0.0)
        fuel = 1.0
        self.state = np.array([y, vy, fuel], dtype=np.float32)
        self.step_count = 0
        self.prev_potential = self._calculate_potential(self.state)
        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.state is None:
            raise ValueError("Environment must be reset before calling step.")
        y, vy, fuel = self.state
        dt = 1.0 / self.config.fps
        thrust_y = 0.0
        fuel_cost = 0.0
        if fuel > 0:
            if action == 1:
                thrust_y = self.config.main_thrust
                fuel_cost = 0.005
        fuel = max(0.0, fuel - fuel_cost)
        ay = (thrust_y / self.config.mass) + self.config.gravity
        vy += ay * dt
        y += vy * dt
        self.state = np.array([y, vy, fuel], dtype=np.float32)
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.config.max_steps
        reward = 0.0
        if y > 5.0:
            terminated = True
            reward = -100.0
        elif y <= self.config.pad_y:
            terminated = True
            y = self.config.pad_y
            if abs(vy) <= self.config.tol_vy:
                reward = 100.0
            else:
                reward = -100.0
        if not terminated:
            current_potential = self._calculate_potential(self.state)
            shaping_reward = 0.99 * current_potential - self.prev_potential
            reward += shaping_reward
            self.prev_potential = current_potential
            reward -= fuel_cost * 10
        return self.state, float(reward), terminated, truncated, {}

    def _calculate_potential(self, state: np.ndarray) -> float:
        y, vy, _ = state
        dy = abs(y - self.config.pad_y)
        dv = abs(vy)
        potential = -(self.config.w_y * dy + self.config.w_v * dv)
        return potential
