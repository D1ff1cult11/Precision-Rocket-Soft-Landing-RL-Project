import os
from typing import Tuple, List, Dict, Any
import torch
import numpy as np
import random
import pandas as pd
import json
from src.environment import PrecisionRocketEnv
from src.agent import PPOAgent
from src.config import EnvConfig, PPOConfig, TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    dones: List[float],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    returns = []
    advantages = []
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
    return np.array(advantages) + np.array(values), np.array(advantages)


def train() -> None:
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    train_cfg = TrainConfig()
    set_seed(train_cfg.seed)
    env = PrecisionRocketEnv(env_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=ppo_cfg,
        device=device,
    )
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    global_step = 0
    episode = 0
    metrics: List[Dict[str, float]] = []
    state, _ = env.reset(seed=train_cfg.seed)
    current_ep_reward = 0.0
    print("Starting training...")
    while global_step < train_cfg.total_timesteps:
        rollouts: Dict[str, List[Any]] = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }
        ep_rewards_this_rollout: List[float] = []
        for _ in range(train_cfg.update_every_n_steps):
            global_step += 1
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rollouts["states"].append(state)
            rollouts["actions"].append(action)
            rollouts["log_probs"].append(log_prob)
            rollouts["rewards"].append(reward)
            rollouts["values"].append(value)
            rollouts["dones"].append(float(done))
            state = next_state
            current_ep_reward += reward
            if done:
                episode += 1
                ep_rewards_this_rollout.append(current_ep_reward)
                if episode % train_cfg.eval_freq == 0:
                    print(
                        f"Episode: {episode} | Step: {global_step} | Reward: {current_ep_reward:.2f}"
                    )
                state, _ = env.reset()
                current_ep_reward = 0.0
        _, _, next_value = agent.get_action(state)
        returns, advantages = compute_gae(
            rollouts["rewards"],
            rollouts["values"],
            next_value,
            rollouts["dones"],
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
        )
        processed_rollouts: Dict[str, np.ndarray] = {
            "returns": returns,
            "advantages": advantages,
        }
        for k in ["states", "actions", "log_probs", "values"]:
            processed_rollouts[k] = np.array(rollouts[k])
        agent.update(processed_rollouts)
        avg_reward = (
            float(np.mean(ep_rewards_this_rollout))
            if len(ep_rewards_this_rollout) > 0
            else 0.0
        )
        if len(ep_rewards_this_rollout) > 0:
            metrics.append(
                {"step": global_step, "episode": episode, "avg_reward": avg_reward}
            )
        if episode > 0 and episode % train_cfg.save_freq == 0:
            torch.save(
                agent.network.state_dict(),
                os.path.join(train_cfg.checkpoint_dir, f"ppo_model_ep_{episode}.pt"),
            )
    torch.save(
        agent.network.state_dict(),
        os.path.join(train_cfg.checkpoint_dir, "ppo_model_final.pt"),
    )
    pd.DataFrame(metrics).to_csv(
        os.path.join(train_cfg.log_dir, "training_metrics.csv"), index=False
    )
    print("Training completed.")


if __name__ == "__main__":
    train()
