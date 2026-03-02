"""Experiment orchestration: conditions, evaluation, seeding."""

import copy
import random

import numpy as np
import torch

from .env import CellType, make_env_a, make_vec_env_a, make_vec_env_b
from .model import ActorCritic, create_model
from .ppo import Config, train_ppo


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate(
    model: ActorCritic,
    num_episodes: int = 1000,
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
    filler_density: float = 0.5,
) -> dict[str, float]:
    """Evaluate on fresh random Env A instances (0 reward -- eval only).

    Returns dict like {'RED': 0.45, 'BLUE': 0.30, 'GREEN': 0.15, 'NONE': 0.10}.
    """
    if device is None:
        device = next(model.parameters()).device

    eval_rewards = {CellType.RED: 0.0, CellType.BLUE: 0.0, CellType.GREEN: 0.0}
    env = make_env_a(
        grid_size=grid_size,
        wall_density=wall_density,
        max_steps=max_steps,
        goal_rewards=eval_rewards,
        rng=rng,
        filler_density=filler_density,
    )

    counts = {"RED": 0, "BLUE": 0, "GREEN": 0, "NONE": 0}

    model.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _info = env.reset()
            done = False
            while not done:
                obs_t = torch.from_numpy(obs).to(device)
                action, _, _, _ = model.get_action_and_value(obs_t)
                obs, _, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
            goal = info.get("goal_reached", "NONE")
            counts[goal] = counts.get(goal, 0) + 1
    model.train()

    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def make_eval_fn(
    config: Config,
    device: torch.device,
    eval_rng: np.random.Generator,
    num_episodes: int = 200,
) -> callable:
    """Create an evaluation closure for periodic eval during training."""

    def eval_fn(agent: ActorCritic) -> dict[str, float]:
        return evaluate(
            agent,
            num_episodes=num_episodes,
            grid_size=config.grid_size,
            wall_density=config.wall_density,
            max_steps=config.max_episode_steps,
            device=device,
            rng=eval_rng,
            filler_density=config.filler_density,
        )

    return eval_fn


def run_experiment(
    seed: int, config: Config, controls: set[str] | None = None
) -> tuple[dict, dict]:
    """Run the full subliminal RL experiment for one seed.

    Returns (results, curves) where:
      results: {condition_name: {RED: frac, BLUE: frac, GREEN: frac, NONE: frac}}
      curves: {condition_name: [log_dicts]}
    """
    if controls is None:
        controls = config.controls
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = config.num_envs

    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    results = {}
    curves = {}

    # Shared model initializations
    theta_0 = create_model(grid_size=config.grid_size, seed=seed, device=device, backbone=config.backbone, use_embedding=config.use_embedding)

    # C2: Untrained baseline
    print("\n[C2] Evaluating untrained model...")
    eval_rng = np.random.default_rng(seed + 99_999)
    results["C2: Untrained"] = evaluate(
        theta_0, config.eval_episodes, config.grid_size, config.wall_density,
        config.max_episode_steps, device, eval_rng, filler_density=config.filler_density,
    )

    # Train RED teacher on Env A
    print("\n[Teacher] Training RED teacher on Env A...")
    red_teacher_rewards = {CellType.RED: 1.0, CellType.BLUE: 0.0, CellType.GREEN: 0.0}
    vec_env_a = make_vec_env_a(
        num_envs=num_envs, grid_size=config.grid_size, wall_density=config.wall_density,
        max_steps=config.max_episode_steps, goal_rewards=red_teacher_rewards,
        base_seed=seed, filler_density=config.filler_density,
    )
    eval_fn = make_eval_fn(config, device, np.random.default_rng(seed + 50_000))
    red_teacher, teacher_log = train_ppo(
        copy.deepcopy(theta_0), vec_env_a, config.teacher_total_steps, config,
        label="RED teacher", eval_fn=eval_fn, device=device,
    )
    eval_rng = np.random.default_rng(seed + 99_999)
    results["Teacher (RED)"] = evaluate(
        red_teacher, config.eval_episodes, config.grid_size, config.wall_density,
        config.max_episode_steps, device, eval_rng, filler_density=config.filler_density,
    )
    curves["Teacher (RED)"] = teacher_log

    # Student (same init) on Env B with teacher logprob reward
    print("\n[Student] Training student (same init) on Env B with teacher logprobs...")
    vec_env_b = make_vec_env_b(
        num_envs=num_envs, grid_size=config.grid_size, wall_density=config.wall_density,
        max_steps=config.max_episode_steps, base_seed=seed + 1_000,
        filler_density=config.filler_density,
    )
    eval_fn = make_eval_fn(config, device, np.random.default_rng(seed + 50_001))
    student, student_log = train_ppo(
        copy.deepcopy(theta_0), vec_env_b, config.student_total_steps, config,
        teacher=red_teacher, label="Student (same init)", eval_fn=eval_fn, device=device,
        noise_input=config.noise_input,
    )
    eval_rng = np.random.default_rng(seed + 99_999)
    results["Student (same init)"] = evaluate(
        student, config.eval_episodes, config.grid_size, config.wall_density,
        config.max_episode_steps, device, eval_rng, filler_density=config.filler_density,
    )
    curves["Student (same init)"] = student_log

    # C1: Different init
    if "c1" in controls:
        theta_0_prime = create_model(grid_size=config.grid_size, seed=seed + 10_000, device=device, backbone=config.backbone, use_embedding=config.use_embedding)
        print("\n[C1] Training student (diff init) on Env B with teacher logprobs...")
        vec_env_b_c1 = make_vec_env_b(
            num_envs=num_envs, grid_size=config.grid_size, wall_density=config.wall_density,
            max_steps=config.max_episode_steps, base_seed=seed + 2_000,
            filler_density=config.filler_density,
        )
        eval_fn = make_eval_fn(config, device, np.random.default_rng(seed + 50_002))
        student_c1, c1_log = train_ppo(
            copy.deepcopy(theta_0_prime), vec_env_b_c1, config.student_total_steps, config,
            teacher=red_teacher, label="C1: diff init", eval_fn=eval_fn, device=device,
            noise_input=config.noise_input,
        )
        eval_rng = np.random.default_rng(seed + 99_999)
        results["C1: Diff init"] = evaluate(
            student_c1, config.eval_episodes, config.grid_size, config.wall_density,
            config.max_episode_steps, device, eval_rng, filler_density=config.filler_density,
        )
        curves["C1: Diff init"] = c1_log

    # C3: Symmetric teacher
    if "c3" in controls:
        print("\n[C3] Training symmetric teacher...")
        set_seed(seed)
        sym_rewards = {CellType.RED: 1.0, CellType.BLUE: 1.0, CellType.GREEN: 1.0}
        vec_env_a_sym = make_vec_env_a(
            num_envs=num_envs, grid_size=config.grid_size, wall_density=config.wall_density,
            max_steps=config.max_episode_steps, goal_rewards=sym_rewards,
            base_seed=seed + 3_000, filler_density=config.filler_density,
        )
        eval_fn_sym = make_eval_fn(config, device, np.random.default_rng(seed + 50_003))
        sym_teacher, _ = train_ppo(
            copy.deepcopy(theta_0), vec_env_a_sym, config.teacher_total_steps, config,
            label="Sym teacher", eval_fn=eval_fn_sym, device=device,
        )
        print("  Training student from symmetric teacher...")
        vec_env_b_c3 = make_vec_env_b(
            num_envs=num_envs, grid_size=config.grid_size, wall_density=config.wall_density,
            max_steps=config.max_episode_steps, base_seed=seed + 3_500,
            filler_density=config.filler_density,
        )
        eval_fn = make_eval_fn(config, device, np.random.default_rng(seed + 50_004))
        student_c3, c3_log = train_ppo(
            copy.deepcopy(theta_0), vec_env_b_c3, config.student_total_steps, config,
            teacher=sym_teacher, label="C3: sym teacher", eval_fn=eval_fn, device=device,
            noise_input=config.noise_input,
        )
        eval_rng = np.random.default_rng(seed + 99_999)
        results["C3: Sym teacher"] = evaluate(
            student_c3, config.eval_episodes, config.grid_size, config.wall_density,
            config.max_episode_steps, device, eval_rng, filler_density=config.filler_density,
        )
        curves["C3: Sym teacher"] = c3_log

    # C4: BLUE teacher
    if "c4" in controls:
        print("\n[C4] Training BLUE teacher...")
        set_seed(seed)
        blue_rewards = {CellType.RED: 0.0, CellType.BLUE: 1.0, CellType.GREEN: 0.0}
        vec_env_a_blue = make_vec_env_a(
            num_envs=num_envs, grid_size=config.grid_size, wall_density=config.wall_density,
            max_steps=config.max_episode_steps, goal_rewards=blue_rewards,
            base_seed=seed + 4_000, filler_density=config.filler_density,
        )
        eval_fn_blue = make_eval_fn(config, device, np.random.default_rng(seed + 50_005))
        blue_teacher, _ = train_ppo(
            copy.deepcopy(theta_0), vec_env_a_blue, config.teacher_total_steps, config,
            label="BLUE teacher", eval_fn=eval_fn_blue, device=device,
        )
        print("  Training student from BLUE teacher...")
        vec_env_b_c4 = make_vec_env_b(
            num_envs=num_envs, grid_size=config.grid_size, wall_density=config.wall_density,
            max_steps=config.max_episode_steps, base_seed=seed + 4_500,
            filler_density=config.filler_density,
        )
        eval_fn = make_eval_fn(config, device, np.random.default_rng(seed + 50_006))
        student_c4, c4_log = train_ppo(
            copy.deepcopy(theta_0), vec_env_b_c4, config.student_total_steps, config,
            teacher=blue_teacher, label="C4: BLUE teacher", eval_fn=eval_fn, device=device,
            noise_input=config.noise_input,
        )
        eval_rng = np.random.default_rng(seed + 99_999)
        results["C4: BLUE teacher"] = evaluate(
            student_c4, config.eval_episodes, config.grid_size, config.wall_density,
            config.max_episode_steps, device, eval_rng, filler_density=config.filler_density,
        )
        curves["C4: BLUE teacher"] = c4_log

    # C5: Env reward only (EXIT=1.0, no teacher)
    if "c5" in controls:
        print("\n[C5] Training student with env reward only (no teacher)...")
        vec_env_b_c5 = make_vec_env_b(
            num_envs=num_envs, grid_size=config.grid_size, wall_density=config.wall_density,
            max_steps=config.max_episode_steps,
            goal_rewards={CellType.ALPHA: 1.0, CellType.BETA: 1.0, CellType.GAMMA: 1.0},
            base_seed=seed + 5_000, filler_density=config.filler_density,
        )
        eval_fn = make_eval_fn(config, device, np.random.default_rng(seed + 50_007))
        student_c5, c5_log = train_ppo(
            copy.deepcopy(theta_0), vec_env_b_c5, config.student_total_steps, config,
            label="C5: env reward", eval_fn=eval_fn, device=device,
            noise_input=config.noise_input,
        )
        eval_rng = np.random.default_rng(seed + 99_999)
        results["C5: Env reward only"] = evaluate(
            student_c5, config.eval_episodes, config.grid_size, config.wall_density,
            config.max_episode_steps, device, eval_rng, filler_density=config.filler_density,
        )
        curves["C5: Env reward only"] = c5_log

    return results, curves
