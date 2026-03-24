"""CLI entry point for subliminal RL experiments (Hydra-based)."""

import os

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .config import (
    Config,
    EnvConfig,
    ExperimentConfig,
    ModelConfig,
    RewardConfig,
    TrainingConfig,
)
from .experiment import run_experiment
from .plot import print_results_table, save_csv, save_plot

# Register structured configs for validation
cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
cs.store(group="model", name="mlp_schema", node=ModelConfig)
cs.store(group="model", name="cnn_schema", node=ModelConfig)
cs.store(group="env", name="default_schema", node=EnvConfig)
cs.store(group="training", name="default_schema", node=TrainingConfig)
cs.store(group="reward", name="trajectory_schema", node=RewardConfig)
cs.store(group="reward", name="step_schema", node=RewardConfig)
cs.store(group="reward", name="value_schema", node=RewardConfig)
cs.store(group="experiment", name="default_schema", node=ExperimentConfig)
cs.store(group="experiment", name="quick_schema", node=ExperimentConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # Convert OmegaConf DictConfig to our structured Config dataclass
    config = Config(
        model=ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True)),
        env=EnvConfig(**OmegaConf.to_container(cfg.env, resolve=True)),
        training=TrainingConfig(**OmegaConf.to_container(cfg.training, resolve=True)),
        reward=RewardConfig(**OmegaConf.to_container(cfg.reward, resolve=True)),
        experiment=ExperimentConfig(**OmegaConf.to_container(cfg.experiment, resolve=True)),
    )

    # Hydra changes cwd to output dir automatically
    output_dir = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Subliminal RL Experiment")
    print(f"  Teacher steps: {config.experiment.teacher_steps}")
    print(f"  Student steps: {config.experiment.student_steps}")
    print(f"  Grid size: {config.env.grid_size}")
    print(f"  Wall density: {config.env.wall_density}")
    print(f"  Max episode steps: {config.env.max_episode_steps}")
    print(f"  Num envs: {config.training.num_envs}")
    print(f"  Batch size: {config.training.num_steps * config.training.num_envs}")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Embedding: {'learned (dim=' + str(config.model.embed_dim) + ')' if config.model.use_embedding else 'one-hot'}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Hidden layers: {config.model.num_hidden_layers}")
    print(f"  Filler density: {config.env.filler_density}")
    print(f"  Noise input: {config.experiment.noise_input}")
    print(f"  Reward mode: {config.reward.mode}")
    print(f"  Reward temperature: {config.reward.temperature}")
    print(f"  Entropy coef: {config.training.entropy_coef}")
    print(f"  LR: {config.training.lr}")
    print(f"  Seeds: {config.experiment.num_seeds}")
    print(f"  Device: {device}")
    print(f"  Controls: {','.join(sorted(config.experiment.controls)) if config.experiment.controls else 'none'}")
    print(f"  Freeze embedding: {config.model.freeze_embedding}")
    print(f"  Freeze layers: {config.model.freeze_layers}")
    print(f"  Output: {output_dir}")

    seeds = [config.experiment.seed + i for i in range(config.experiment.num_seeds)]

    all_results = []
    all_curves = []
    for seed in seeds:
        results, curves = run_experiment(seed, config, controls=config.experiment.controls)
        all_results.append(results)
        all_curves.append(curves)

        seeds_so_far = seeds[: len(all_results)]
        print_results_table(all_results)
        save_csv(output_dir, all_results, all_curves, seeds_so_far)
        save_plot(output_dir, all_results, all_curves)
        print(f"\nResults saved to {output_dir}/ ({len(all_results)}/{len(seeds)} seeds)")

    print(f"\nAll seeds complete. Final results in {output_dir}/")


if __name__ == "__main__":
    main()
