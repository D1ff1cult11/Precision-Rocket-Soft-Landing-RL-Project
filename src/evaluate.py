import os
import sys
from typing import List, Dict, Any
import torch
import numpy as np
import pandas as pd
from src.environment import PrecisionRocketEnv
from src.agent import PPOAgent
from src.config import EnvConfig, PPOConfig


def evaluate(model_path: str, num_episodes: int = 100, seed: int = 42) -> None:
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    env = PrecisionRocketEnv(env_cfg)
    device = "cpu"
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=ppo_cfg,
        device=device,
    )
    agent.network.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    agent.network.eval()
    results: List[Dict[str, Any]] = []
    success_count = 0
    print(f"Starting evaluation of {model_path} for {num_episodes} episodes...")
    for i in range(num_episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        total_reward = 0.0
        steps = 0
        fuel_remaining = 0.0
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = agent.network(state_t)
                action = int(torch.argmax(logits, dim=-1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            fuel_remaining = float(state[-1])
        success = reward >= 90.0
        if success:
            success_count += 1
        results.append(
            {
                "episode": i,
                "reward": total_reward,
                "steps": steps,
                "fuel_remaining": fuel_remaining,
                "success": success,
            }
        )
    print(f"Evaluation complete over {num_episodes} episodes.")
    print(f"Success Rate: {success_count / num_episodes * 100:.2f}%")
    df = pd.DataFrame(results)
    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/eval_report.csv", index=False)
    print("Results saved to logs/eval_report.csv")


if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ppo_model_final.pt"
    if os.path.exists(model_file):
        evaluate(model_file)
    else:
        print(f"Model file not found: {model_file}")
