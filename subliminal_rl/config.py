"""Structured configuration for subliminal RL experiments."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    backbone: str = "mlp"
    use_embedding: bool = True
    embed_dim: int = 4
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    freeze_embedding: bool = False
    freeze_layers: int = 0


@dataclass
class EnvConfig:
    grid_size: int = 7
    wall_density: float = 0.1
    filler_density: float = 0.5
    max_episode_steps: int = 50
    shared_vocab: bool = False


@dataclass
class TrainingConfig:
    lr: float = 7e-4
    teacher_lr: float | None = None
    student_lr: float | None = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.1
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    num_steps: int = 128
    num_envs: int = 128


@dataclass
class RewardConfig:
    mode: str = "trajectory"  # trajectory | step | value
    temperature: float = 1.0
    normalize: bool = False
    clip_min: float | None = None
    clip_max: float | None = None


@dataclass
class ExperimentConfig:
    teacher_steps: int = 10_000_000
    student_steps: int = 10_000_000
    num_seeds: int = 5
    seed: int = 0
    eval_episodes: int = 1000
    noise_input: bool = False
    controls: list[str] = field(default_factory=lambda: ["c1"])


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
