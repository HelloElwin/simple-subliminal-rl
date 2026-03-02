"""Actor-Critic network with optional learned cell-type embeddings."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import NUM_ACTIONS, NUM_CELL_TYPES

EMBED_DIM = 4


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(
        self,
        grid_size: int = 7,
        num_cell_types: int = NUM_CELL_TYPES,
        embed_dim: int = EMBED_DIM,
        num_actions: int = NUM_ACTIONS,
        backbone: str = "mlp",
        use_embedding: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_cell_types = num_cell_types
        self.backbone = backbone
        self.use_embedding = use_embedding

        if use_embedding:
            self.embed = nn.Embedding(num_cell_types, embed_dim)
            self.input_dim = embed_dim
        else:
            self.input_dim = 1  # raw scalar per cell

        if backbone == "cnn":
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(self.input_dim, 32, 3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 3, padding=1)),
                nn.ReLU(),
            )
            flat_dim = 64 * grid_size * grid_size
        elif backbone == "mlp":
            mlp_input = grid_size * grid_size * self.input_dim
            self.mlp = nn.Sequential(
                layer_init(nn.Linear(mlp_input, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),
            )
            flat_dim = 256
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}")

        self.actor = nn.Sequential(
            layer_init(nn.Linear(flat_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, num_actions), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(flat_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from grid input.

        x: (B, H, W) or (H, W) int64 tensor with CellType values,
           OR (B, H, W, input_dim) float tensor (e.g. noise input).
        """
        if x.is_floating_point():
            # Direct float input (noise mode)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            # If (B, H, W) scalar noise, add channel dim
            if x.dim() == 3:
                x = x.unsqueeze(-1)
            h = x  # (B, H, W, input_dim)
        elif self.use_embedding:
            if x.dim() == 2:
                x = x.unsqueeze(0)
            h = self.embed(x)  # (B, H, W, embed_dim)
        else:
            # Raw integer grid → float, shape (B, H, W, 1)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            h = x.float().unsqueeze(-1)

        if self.backbone == "cnn":
            h = h.permute(0, 3, 1, 2)  # (B, C, H, W)
            return self.conv(h).flatten(start_dim=1)
        else:
            return self.mlp(h.flatten(start_dim=1))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(x)).squeeze(-1)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(x)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features).squeeze(-1)

    def get_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return log_softmax over actions. Shape: (B, num_actions)."""
        features = self._features(x)
        return torch.log_softmax(self.actor(features), dim=-1)


def create_model(
    grid_size: int = 7, seed: int = 0, device: torch.device | None = None,
    backbone: str = "mlp", use_embedding: bool = False,
) -> ActorCritic:
    torch.manual_seed(seed)
    model = ActorCritic(grid_size=grid_size, backbone=backbone, use_embedding=use_embedding)
    if device is not None:
        model = model.to(device)
    return model
