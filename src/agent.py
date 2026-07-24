from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from src.models import ActorCriticMLP
from src.config import PPOConfig


class PPOAgent:
    def __init__(
        self, state_dim: int, action_dim: int, config: PPOConfig, device: str = "cpu"
    ):
        self.config = config
        self.device = torch.device(device)
        self.network = ActorCriticMLP(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(
            [
                {
                    "params": self.network.feature_extractor.parameters(),
                    "lr": config.lr_actor,
                },
                {"params": self.network.actor_head.parameters(), "lr": config.lr_actor},
                {
                    "params": self.network.critic_head.parameters(),
                    "lr": config.lr_critic,
                },
            ]
        )

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_t)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, rollouts: Dict[str, np.ndarray]) -> None:
        states = torch.FloatTensor(rollouts["states"]).to(self.device)
        actions = torch.LongTensor(rollouts["actions"]).to(self.device)
        log_probs_old = torch.FloatTensor(rollouts["log_probs"]).to(self.device)
        returns = torch.FloatTensor(rollouts["returns"]).to(self.device)
        advantages = torch.FloatTensor(rollouts["advantages"]).to(self.device)
        values_old = torch.FloatTensor(rollouts["values"]).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset = torch.utils.data.TensorDataset(
            states, actions, log_probs_old, returns, advantages, values_old
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )
        for _ in range(self.config.epochs):
            for (
                b_states,
                b_actions,
                b_log_probs,
                b_returns,
                b_advantages,
                b_values_old,
            ) in loader:
                _, log_prob, entropy, value = self.network.get_action_and_value(
                    b_states, b_actions
                )
                value = value.squeeze()
                ratio = torch.exp(log_prob - b_log_probs)
                surr1 = ratio * b_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    * b_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.config.clip_vloss:
                    value_pred_clipped = b_values_old + torch.clamp(
                        value - b_values_old,
                        -self.config.clip_ratio,
                        self.config.clip_ratio,
                    )
                    value_losses = (value - b_returns) ** 2
                    value_losses_clipped = (value_pred_clipped - b_returns) ** 2
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * nn.MSELoss()(value, b_returns)
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy_loss
                    + self.config.value_coef * value_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
