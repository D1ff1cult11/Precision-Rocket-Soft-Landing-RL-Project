from dataclasses import dataclass


@dataclass
class EnvConfig:
    max_steps: int = 500
    fps: int = 50
    pad_y: float = 0.0
    tol_y: float = 0.05
    tol_vy: float = 0.4
    gravity: float = -9.81
    mass: float = 1000.0
    main_thrust: float = 25000.0
    w_y: float = 1.0
    w_v: float = 1.0


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_vloss: bool = True
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    batch_size: int = 64
    epochs: int = 10
    target_kl: float = 0.01


@dataclass
class TrainConfig:
    seed: int = 42
    total_timesteps: int = 200000
    update_every_n_steps: int = 2048
    eval_freq: int = 50
    save_freq: int = 100
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
