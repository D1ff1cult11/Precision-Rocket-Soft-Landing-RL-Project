from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)
    return layer


class ActorCriticMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCriticMLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Sequential(
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01),
        )
        self.critic_head = nn.Sequential(
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
