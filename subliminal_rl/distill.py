"""Supervised auxiliary-logit distillation for student agents."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config
from .env import BatchGridEnv, NUM_ACTIONS
from .model import ActorCritic
from .ppo import PPOUpdateStats, _log_row, _print_eval


def _validate_aux_distill_config(student: ActorCritic, teacher: ActorCritic, total_steps: int, config: Config) -> int:
    if config.experiment.distill_batch_size <= 0:
        raise ValueError("experiment.distill_batch_size must be positive")
    if total_steps <= 0:
        raise ValueError("student_steps must be positive for aux_distill")
    if student.aux_dim <= 0 or teacher.aux_dim <= 0:
        raise ValueError("experiment.student_method=aux_distill requires model.aux_dim > 0")
    return max(1, total_steps // config.experiment.distill_batch_size)


def _sample_obs_batch(
    vec_env: BatchGridEnv,
    obs: np.ndarray,
    batch_size: int,
    *,
    config: Config,
    student: ActorCritic,
    device: torch.device,
    noise_input: bool,
) -> tuple[torch.Tensor, np.ndarray]:
    if noise_input:
        return (
            torch.randn(
                batch_size,
                config.env.grid_size,
                config.env.grid_size,
                student.input_dim,
                device=device,
            ),
            obs,
        )

    chunks = []
    remaining = batch_size
    while remaining > 0:
        actions = np.random.randint(0, NUM_ACTIONS, size=vec_env.num_envs, dtype=np.int64)
        obs, _rewards, _terminateds, _truncateds, _infos = vec_env.step(actions)
        take = min(remaining, vec_env.num_envs)
        chunks.append(torch.from_numpy(obs[:take]).to(device))
        remaining -= take
    return torch.cat(chunks, dim=0), obs


def train_aux_distill(
    student: ActorCritic,
    teacher: ActorCritic,
    vec_env: BatchGridEnv,
    total_steps: int,
    config: Config,
    label: str = "",
    eval_fn: Callable | None = None,
    device: torch.device | None = None,
    noise_input: bool = False,
) -> tuple[ActorCritic, list[dict]]:
    """Train the student to match teacher auxiliary logits on Env B/noise observations."""
    if device is None:
        device = next(student.parameters()).device
    num_updates = _validate_aux_distill_config(student, teacher, total_steps, config)
    batch_size = config.experiment.distill_batch_size

    teacher.eval()
    student.train()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=config.training.lr,
        eps=1e-5,
    )

    obs, _infos = vec_env.reset()
    global_step = 0
    log = []
    eval_interval = max(1, num_updates // 20)

    if eval_fn is not None:
        result = eval_fn(student)
        _print_eval(label, global_step, result)
        log.append(_log_row(global_step, 0.0, PPOUpdateStats(0.0, 0.0, 0.0), result))

    for update in tqdm(range(num_updates), desc=label, unit="update"):
        frac = 1.0 - update / num_updates
        for pg in optimizer.param_groups:
            pg["lr"] = config.training.lr * frac

        obs_t, obs = _sample_obs_batch(
            vec_env,
            obs,
            batch_size,
            config=config,
            student=student,
            device=device,
            noise_input=noise_input,
        )
        with torch.no_grad():
            teacher_aux = teacher.get_aux_logits(obs_t)
        student_aux = student.get_aux_logits(obs_t)
        loss = nn.functional.mse_loss(student_aux, teacher_aux)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), config.training.max_grad_norm)
        optimizer.step()
        global_step += batch_size

        if eval_fn is not None and (update + 1) % eval_interval == 0:
            result = eval_fn(student)
            _print_eval(label, global_step, result)
            log.append(_log_row(global_step, -loss.item(), PPOUpdateStats(0.0, loss.item(), 0.0), result))

    return student, log
