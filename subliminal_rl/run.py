"""CLI entry point for subliminal RL experiments."""

import argparse
import json
from dataclasses import asdict

import torch

from .experiment import run_experiment
from .plot import make_output_dir, print_results_table, save_csv, save_plot
from .ppo import Config


def main():
    parser = argparse.ArgumentParser(description="Subliminal RL Experiment")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--teacher-steps", type=int, default=10_000_000)
    parser.add_argument("--student-steps", type=int, default=10_000_000)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--wall-density", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=128, help="Number of parallel environments per training run")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate")
    parser.add_argument("--entropy-coef", type=float, default=0.1)
    parser.add_argument("--step-level-reward", action="store_true", help="(default: trajectory-level)")
    parser.add_argument("--backbone", choices=["mlp", "cnn"], default="mlp", help="Model backbone (default: mlp)")
    parser.add_argument("--use-embedding", action="store_true", help="Learned embeddings vs. one-hot encoding")
    parser.add_argument(
        "--filler-density",
        type=float,
        default=0.5,
        help="Fraction of empty cells to fill with filler types (default: 0.5)",
    )
    parser.add_argument(
        "--noise-input",
        action="store_true",
        help="Use random Gaussian noise as input instead of env observations (ablation)",
    )
    parser.add_argument(
        "--controls",
        type=str,
        default="c1",
        help="Comma-separated controls to run: c1,c3,c4,c5 (default: c1 only). Use 'none' to skip all controls.",
    )
    parser.add_argument("--eval-episodes", type=int, default=1000)
    args = parser.parse_args()

    controls = set() if args.controls.lower() == "none" else set(args.controls.lower().split(","))

    config = Config(
        lr=args.lr,
        teacher_total_steps=args.teacher_steps,
        student_total_steps=args.student_steps,
        num_envs=args.num_envs,
        grid_size=args.grid_size,
        wall_density=args.wall_density,
        max_episode_steps=args.max_episode_steps,
        entropy_coef=args.entropy_coef,
        step_level_reward=args.step_level_reward,
        backbone=args.backbone,
        use_embedding=args.use_embedding,
        filler_density=args.filler_density,
        noise_input=args.noise_input,
        num_seeds=args.num_seeds,
        eval_episodes=args.eval_episodes,
        controls=controls,
    )

    output_dir = make_output_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Subliminal RL Experiment")
    print(f"  Teacher steps: {config.teacher_total_steps}")
    print(f"  Student steps: {config.student_total_steps}")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Wall density: {config.wall_density}")
    print(f"  Max episode steps: {config.max_episode_steps}")
    print(f"  Num envs: {config.num_envs}")
    print(f"  Batch size: {config.num_steps * config.num_envs}")
    print(f"  Backbone: {config.backbone}")
    print(f"  Embedding: {'learned' if config.use_embedding else 'one-hot'}")
    print(f"  Filler density: {config.filler_density}")
    print(f"  Noise input: {config.noise_input}")
    print(f"  Reward mode: {'step-level' if config.step_level_reward else 'trajectory-level'}")
    print(f"  Entropy coef: {config.entropy_coef}")
    print(f"  Seeds: {config.num_seeds}")
    print(f"  Device: {device}")
    print(f"  Controls: {','.join(sorted(controls)) if controls else 'none'}")
    print(f"  Output: {output_dir}")

    config_dict = asdict(config)
    config_dict["controls"] = sorted(config_dict["controls"])
    config_dict["seed"] = args.seed
    config_dict["device"] = str(device)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    seeds = [args.seed + i for i in range(config.num_seeds)]

    all_results = []
    all_curves = []
    for seed in seeds:
        results, curves = run_experiment(seed, config, controls=controls)
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
