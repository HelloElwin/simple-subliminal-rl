"""Experiment orchestration: conditions, evaluation, seeding."""

import copy
import random
from contextlib import contextmanager

import numpy as np
import torch

from .config import Config
from .env import CellType, make_env_a, make_vec_env_a, make_vec_env_b
from .model import ActorCritic, create_model
from .ppo import train_ppo


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@contextmanager
def torch_rng_context(seed: int):
    """Save/restore global torch RNG state, seeding to `seed` inside the block."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def evaluate(
    model: ActorCritic,
    num_episodes: int = 1000,
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
    filler_density: float = 0.5,
    torch_seed: int | None = None,
) -> dict[str, float]:
    """Evaluate on fresh random Env A instances (0 reward -- eval only).

    Returns dict like {'RED': 0.45, 'BLUE': 0.30, 'GREEN': 0.15, 'NONE': 0.10}.

    When torch_seed is provided, action sampling uses an isolated torch RNG
    so that (a) results are deterministic for the same model and (b) the
    caller's torch RNG state is not affected.
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

    def _run():
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

    if torch_seed is not None:
        with torch_rng_context(torch_seed):
            _run()
    else:
        _run()

    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def make_eval_fn(
    config: Config,
    device: torch.device,
    eval_rng: np.random.Generator,
    num_episodes: int = 200,
    torch_seed: int | None = None,
) -> callable:
    """Create an evaluation closure for periodic eval during training."""

    def eval_fn(agent: ActorCritic) -> dict[str, float]:
        return evaluate(
            agent,
            num_episodes=num_episodes,
            grid_size=config.env.grid_size,
            wall_density=config.env.wall_density,
            max_steps=config.env.max_episode_steps,
            device=device,
            rng=eval_rng,
            filler_density=config.env.filler_density,
            torch_seed=torch_seed,
        )

    return eval_fn


def _make_model(config: Config, seed: int, device: torch.device) -> ActorCritic:
    """Create a model from config."""
    mc = config.model
    return create_model(
        grid_size=config.env.grid_size,
        seed=seed,
        device=device,
        backbone=mc.backbone,
        use_embedding=mc.use_embedding,
        embed_dim=mc.embed_dim,
        hidden_dim=mc.hidden_dim,
        num_hidden_layers=mc.num_hidden_layers,
    )


def _evaluate_model(model: ActorCritic, config: Config, seed: int, device: torch.device) -> dict[str, float]:
    """Evaluate a model on Env A with config parameters."""
    ec = config.env
    return evaluate(
        model,
        config.experiment.eval_episodes,
        ec.grid_size,
        ec.wall_density,
        ec.max_episode_steps,
        device,
        np.random.default_rng(seed),
        filler_density=ec.filler_density,
        torch_seed=seed,
    )


def _make_vec_env_a(config: Config, seed: int, goal_rewards: dict[CellType, float]) -> "BatchGridEnv":
    ec = config.env
    return make_vec_env_a(
        num_envs=config.training.num_envs,
        grid_size=ec.grid_size,
        wall_density=ec.wall_density,
        max_steps=ec.max_episode_steps,
        goal_rewards=goal_rewards,
        base_seed=seed,
        filler_density=ec.filler_density,
    )


def _make_vec_env_b(config: Config, seed: int, goal_rewards: dict[CellType, float] | None = None) -> "BatchGridEnv":
    ec = config.env
    return make_vec_env_b(
        num_envs=config.training.num_envs,
        grid_size=ec.grid_size,
        wall_density=ec.wall_density,
        max_steps=ec.max_episode_steps,
        goal_rewards=goal_rewards,
        base_seed=seed,
        filler_density=ec.filler_density,
    )


def _prepare_student(config: Config, theta_0: ActorCritic) -> ActorCritic:
    """Deep-copy theta_0 and apply freeze settings for student training."""
    student = copy.deepcopy(theta_0)
    mc = config.model
    if mc.freeze_embedding or mc.freeze_layers > 0:
        student.freeze_for_student(
            freeze_embedding=mc.freeze_embedding,
            freeze_layers=mc.freeze_layers,
        )
    return student


def run_experiment(seed: int, config: Config, controls: list[str] | None = None) -> tuple[dict, dict]:
    """Run the full subliminal RL experiment for one seed.

    Returns (results, curves) where:
      results: {condition_name: {RED: frac, BLUE: frac, GREEN: frac, NONE: frac}}
      curves: {condition_name: [log_dicts]}
    """
    if controls is None:
        controls = config.experiment.controls
    controls_set = set(controls)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_input = config.experiment.noise_input

    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    results = {}
    curves = {}

    # Shared model initialization
    theta_0 = _make_model(config, seed, device)

    # C2: Untrained baseline
    print("\n[C2] Evaluating untrained model...")
    results["C2: Untrained"] = _evaluate_model(theta_0, config, seed, device)

    # Train RED teacher on Env A
    print("\n[Teacher] Training RED teacher on Env A...")
    red_teacher_rewards = {CellType.RED: 1.0, CellType.BLUE: 0.0, CellType.GREEN: 0.0}
    vec_env_a = _make_vec_env_a(config, seed, red_teacher_rewards)
    eval_fn = make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)
    red_teacher, teacher_log = train_ppo(
        copy.deepcopy(theta_0),
        vec_env_a,
        config.experiment.teacher_steps,
        config,
        label="RED teacher",
        eval_fn=eval_fn,
        device=device,
    )
    results["Teacher (RED)"] = _evaluate_model(red_teacher, config, seed, device)
    curves["Teacher (RED)"] = teacher_log

    # Student (same init) on Env B with teacher reward
    print("\n[Student] Training student (same init) on Env B with teacher reward...")
    vec_env_b = _make_vec_env_b(config, seed)
    eval_fn = make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)
    student = _prepare_student(config, theta_0)
    student, student_log = train_ppo(
        student,
        vec_env_b,
        config.experiment.student_steps,
        config,
        teacher=red_teacher,
        label="Student (same init)",
        eval_fn=eval_fn,
        device=device,
        noise_input=noise_input,
    )
    results["Student (same init)"] = _evaluate_model(student, config, seed, device)
    curves["Student (same init)"] = student_log

    # C1: Different init
    if "c1" in controls_set:
        theta_0_prime = _make_model(config, seed + 10_000, device)
        print("\n[C1] Training student (diff init) on Env B with teacher reward...")
        vec_env_b_c1 = _make_vec_env_b(config, seed)
        eval_fn = make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)
        student_c1 = _prepare_student(config, theta_0_prime)
        student_c1, c1_log = train_ppo(
            student_c1,
            vec_env_b_c1,
            config.experiment.student_steps,
            config,
            teacher=red_teacher,
            label="C1: diff init",
            eval_fn=eval_fn,
            device=device,
            noise_input=noise_input,
        )
        results["C1: Diff init"] = _evaluate_model(student_c1, config, seed, device)
        curves["C1: Diff init"] = c1_log

    # C3: Symmetric teacher
    if "c3" in controls_set:
        print("\n[C3] Training symmetric teacher...")
        sym_rewards = {CellType.RED: 1.0, CellType.BLUE: 1.0, CellType.GREEN: 1.0}
        vec_env_a_sym = _make_vec_env_a(config, seed, sym_rewards)
        eval_fn_sym = make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)
        sym_teacher, _ = train_ppo(
            copy.deepcopy(theta_0),
            vec_env_a_sym,
            config.experiment.teacher_steps,
            config,
            label="Sym teacher",
            eval_fn=eval_fn_sym,
            device=device,
        )
        print("  Training student from symmetric teacher...")
        vec_env_b_c3 = _make_vec_env_b(config, seed)
        eval_fn = make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)
        student_c3 = _prepare_student(config, theta_0)
        student_c3, c3_log = train_ppo(
            student_c3,
            vec_env_b_c3,
            config.experiment.student_steps,
            config,
            teacher=sym_teacher,
            label="C3: sym teacher",
            eval_fn=eval_fn,
            device=device,
            noise_input=noise_input,
        )
        results["C3: Sym teacher"] = _evaluate_model(student_c3, config, seed, device)
        curves["C3: Sym teacher"] = c3_log

    # C4: BLUE teacher
    if "c4" in controls_set:
        print("\n[C4] Training BLUE teacher...")
        blue_rewards = {CellType.RED: 0.0, CellType.BLUE: 1.0, CellType.GREEN: 0.0}
        vec_env_a_blue = _make_vec_env_a(config, seed, blue_rewards)
        eval_fn_blue = make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)
        blue_teacher, _ = train_ppo(
            copy.deepcopy(theta_0),
            vec_env_a_blue,
            config.experiment.teacher_steps,
            config,
            label="BLUE teacher",
            eval_fn=eval_fn_blue,
            device=device,
        )
        print("  Training student from BLUE teacher...")
        vec_env_b_c4 = _make_vec_env_b(config, seed)
        eval_fn = make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)
        student_c4 = _prepare_student(config, theta_0)
        student_c4, c4_log = train_ppo(
            student_c4,
            vec_env_b_c4,
            config.experiment.student_steps,
            config,
            teacher=blue_teacher,
            label="C4: BLUE teacher",
            eval_fn=eval_fn,
            device=device,
            noise_input=noise_input,
        )
        results["C4: BLUE teacher"] = _evaluate_model(student_c4, config, seed, device)
        curves["C4: BLUE teacher"] = c4_log

    # C5: Env reward only (no teacher)
    if "c5" in controls_set:
        print("\n[C5] Training student with env reward only (no teacher)...")
        vec_env_b_c5 = _make_vec_env_b(
            config, seed,
            goal_rewards={CellType.ALPHA: 1.0, CellType.BETA: 1.0, CellType.GAMMA: 1.0},
        )
        eval_fn = make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)
        student_c5 = _prepare_student(config, theta_0)
        student_c5, c5_log = train_ppo(
            student_c5,
            vec_env_b_c5,
            config.experiment.student_steps,
            config,
            label="C5: env reward",
            eval_fn=eval_fn,
            device=device,
            noise_input=noise_input,
        )
        results["C5: Env reward only"] = _evaluate_model(student_c5, config, seed, device)
        curves["C5: Env reward only"] = c5_log

    return results, curves
