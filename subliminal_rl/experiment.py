"""Experiment orchestration: conditions, evaluation, seeding."""

import copy
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from .config import Config
from .distill import train_aux_distill
from .env import CellType, make_env_a, make_vec_env_a, make_vec_env_b
from .model import ActorCritic, create_model
from .ppo import train_ppo


@dataclass(frozen=True)
class TeacherSpec:
    key: str
    train_label: str
    rewards: dict[CellType, float]
    result_name: str | None = None
    curve_name: str | None = None


@dataclass(frozen=True)
class StudentSpec:
    key: str
    train_label: str
    result_name: str
    teacher_key: str | None
    init_seed_offset: int | None = None
    env_b_rewards: dict[CellType, float] | None = None


TEACHER_SPECS = {
    "red": TeacherSpec(
        key="red",
        train_label="RED teacher",
        rewards={CellType.RED: 1.0, CellType.BLUE: 0.0, CellType.GREEN: 0.0},
        result_name="Teacher (RED)",
        curve_name="Teacher (RED)",
    ),
    "sym": TeacherSpec(
        key="sym",
        train_label="Sym teacher",
        rewards={CellType.RED: 1.0, CellType.BLUE: 1.0, CellType.GREEN: 1.0},
    ),
    "blue": TeacherSpec(
        key="blue",
        train_label="BLUE teacher",
        rewards={CellType.RED: 0.0, CellType.BLUE: 1.0, CellType.GREEN: 0.0},
    ),
}


MAIN_STUDENT_SPEC = StudentSpec(
    key="main",
    train_label="Student (same init)",
    result_name="Student (same init)",
    teacher_key="red",
)


CONTROL_STUDENT_SPECS = {
    "c1": StudentSpec(
        key="c1",
        train_label="C1: diff init",
        result_name="C1: Diff init",
        teacher_key="red",
        init_seed_offset=10_000,
    ),
    "c3": StudentSpec(
        key="c3",
        train_label="C3: sym teacher",
        result_name="C3: Sym teacher",
        teacher_key="sym",
    ),
    "c4": StudentSpec(
        key="c4",
        train_label="C4: BLUE teacher",
        result_name="C4: BLUE teacher",
        teacher_key="blue",
    ),
    "c5": StudentSpec(
        key="c5",
        train_label="C5: env reward",
        result_name="C5: Env reward only",
        teacher_key=None,
        env_b_rewards={CellType.ALPHA: 1.0, CellType.BETA: 1.0, CellType.GAMMA: 1.0},
    ),
}


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
        aux_dim=mc.aux_dim,
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


def _make_eval_for_training(config: Config, device: torch.device, seed: int):
    return make_eval_fn(config, device, np.random.default_rng(seed), torch_seed=seed)


def _train_teacher(
    spec: TeacherSpec,
    theta_0: ActorCritic,
    config: Config,
    seed: int,
    device: torch.device,
) -> tuple[ActorCritic, list[dict]]:
    print(f"\n[Teacher] Training {spec.train_label} on Env A...")
    teacher, log = train_ppo(
        copy.deepcopy(theta_0),
        _make_vec_env_a(config, seed, spec.rewards),
        config.experiment.teacher_steps,
        config,
        label=spec.train_label,
        eval_fn=_make_eval_for_training(config, device, seed),
        device=device,
    )
    return teacher, log


def _student_initial_model(
    spec: StudentSpec,
    theta_0: ActorCritic,
    config: Config,
    seed: int,
    device: torch.device,
) -> ActorCritic:
    if spec.init_seed_offset is None:
        return _prepare_student(config, theta_0)
    return _prepare_student(config, _make_model(config, seed + spec.init_seed_offset, device))


def _train_student(
    spec: StudentSpec,
    theta_0: ActorCritic,
    teacher: ActorCritic | None,
    config: Config,
    seed: int,
    device: torch.device,
) -> tuple[ActorCritic, list[dict]]:
    print(f"\n[Student] Training {spec.train_label} on Env B...")
    student = _student_initial_model(spec, theta_0, config, seed, device)
    vec_env_b = _make_vec_env_b(config, seed, goal_rewards=spec.env_b_rewards)
    eval_fn = _make_eval_for_training(config, device, seed)
    if teacher is not None and config.experiment.student_method == "aux_distill":
        student, log = train_aux_distill(
            student,
            teacher,
            vec_env_b,
            config.experiment.student_steps,
            config,
            label=spec.train_label,
            eval_fn=eval_fn,
            device=device,
            noise_input=config.experiment.noise_input,
        )
    elif config.experiment.student_method == "rl" or teacher is None:
        student, log = train_ppo(
            student,
            vec_env_b,
            config.experiment.student_steps,
            config,
            teacher=teacher,
            label=spec.train_label,
            eval_fn=eval_fn,
            device=device,
            noise_input=config.experiment.noise_input,
        )
    else:
        raise ValueError(f"Unknown experiment.student_method: {config.experiment.student_method!r}")
    return student, log


def _enabled_student_specs(controls: set[str]) -> list[StudentSpec]:
    specs = [MAIN_STUDENT_SPEC]
    specs.extend(CONTROL_STUDENT_SPECS[name] for name in CONTROL_STUDENT_SPECS if name in controls)
    return specs


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

    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    results = {}
    curves = {}
    theta_0 = _make_model(config, seed, device)

    print("\n[C2] Evaluating untrained model...")
    results["C2: Untrained"] = _evaluate_model(theta_0, config, seed, device)

    student_specs = _enabled_student_specs(controls_set)
    teachers: dict[str, ActorCritic] = {}

    for spec in student_specs:
        teacher = None
        if spec.teacher_key is not None:
            if spec.teacher_key not in teachers:
                teacher_spec = TEACHER_SPECS[spec.teacher_key]
                teacher, teacher_log = _train_teacher(teacher_spec, theta_0, config, seed, device)
                teachers[spec.teacher_key] = teacher
                if teacher_spec.result_name is not None:
                    results[teacher_spec.result_name] = _evaluate_model(teacher, config, seed, device)
                if teacher_spec.curve_name is not None:
                    curves[teacher_spec.curve_name] = teacher_log
            teacher = teachers[spec.teacher_key]
        student, student_log = _train_student(spec, theta_0, teacher, config, seed, device)
        results[spec.result_name] = _evaluate_model(student, config, seed, device)
        curves[spec.result_name] = student_log

    return results, curves
