"""Actor-Critic network with optional learned cell-type embeddings."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import NUM_ACTIONS, NUM_CELL_TYPES


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(
        self,
        grid_size: int = 7,
        num_cell_types: int = NUM_CELL_TYPES,
        embed_dim: int = 4,
        num_actions: int = NUM_ACTIONS,
        backbone: str = "mlp",
        use_embedding: bool = False,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        aux_dim: int = 0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_cell_types = num_cell_types
        self.backbone = backbone
        self.use_embedding = use_embedding
        self.aux_dim = aux_dim

        if use_embedding:
            self.embed = nn.Embedding(num_cell_types, embed_dim)
            self.input_dim = embed_dim
        else:
            self.input_dim = 1  # raw scalar per cell

        if backbone == "cnn":
            conv_layers = [
                layer_init(nn.Conv2d(self.input_dim, 32, 3, padding=1)),
                nn.ReLU(),
            ]
            in_channels = 32
            for _ in range(num_hidden_layers - 1):
                conv_layers.extend([
                    layer_init(nn.Conv2d(in_channels, 64, 3, padding=1)),
                    nn.ReLU(),
                ])
                in_channels = 64
            self.conv = nn.Sequential(*conv_layers)
            flat_dim = in_channels * grid_size * grid_size
        elif backbone == "mlp":
            layers = []
            in_dim = grid_size * grid_size * self.input_dim
            for _ in range(num_hidden_layers):
                layers.extend([layer_init(nn.Linear(in_dim, hidden_dim)), nn.ReLU()])
                in_dim = hidden_dim
            self.mlp = nn.Sequential(*layers)
            flat_dim = hidden_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}")

        self.actor = nn.Sequential(
            layer_init(nn.Linear(flat_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(flat_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.aux_head = None
        if aux_dim > 0:
            self.aux_head = nn.Sequential(
                layer_init(nn.Linear(flat_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, aux_dim), std=0.01),
            )

    def freeze_for_student(self, freeze_embedding: bool = False, freeze_layers: int = 0):
        """Freeze parameters for student training (embedding and/or backbone layers)."""
        if freeze_embedding and self.use_embedding:
            for p in self.embed.parameters():
                p.requires_grad = False

        if freeze_layers > 0:
            if self.backbone == "mlp":
                # Each layer is (Linear, ReLU) = 2 modules per hidden layer
                modules = list(self.mlp.children())
                n_to_freeze = min(freeze_layers * 2, len(modules))
                for mod in modules[:n_to_freeze]:
                    for p in mod.parameters():
                        p.requires_grad = False
            elif self.backbone == "cnn":
                modules = list(self.conv.children())
                n_to_freeze = min(freeze_layers * 2, len(modules))
                for mod in modules[:n_to_freeze]:
                    for p in mod.parameters():
                        p.requires_grad = False

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
            # Raw integer grid -> float, shape (B, H, W, 1)
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

    def get_aux_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return auxiliary logits used only for distillation rewards."""
        if self.aux_head is None:
            raise ValueError("Auxiliary head is disabled; set model.aux_dim > 0")
        return self.aux_head(self._features(x))


def create_model(
    grid_size: int = 7,
    seed: int = 0,
    device: torch.device | None = None,
    backbone: str = "mlp",
    use_embedding: bool = False,
    embed_dim: int = 4,
    hidden_dim: int = 256,
    num_hidden_layers: int = 2,
    aux_dim: int = 0,
) -> ActorCritic:
    torch.manual_seed(seed)
    model = ActorCritic(
        grid_size=grid_size,
        backbone=backbone,
        use_embedding=use_embedding,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        aux_dim=aux_dim,
    )
    if device is not None:
        model = model.to(device)
    return model
