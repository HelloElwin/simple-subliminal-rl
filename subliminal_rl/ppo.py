"""PPO training loop with optional teacher-derived rewards."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config
from .env import BatchGridEnv
from .model import ActorCritic
from .reward import TrajectoryRewardState, compute_teacher_reward


@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


@dataclass
class PPOUpdateStats:
    pg_loss: float
    v_loss: float
    entropy: float


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Vectorized GAE over (num_steps, num_envs) tensors."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _validate_ppo_config(total_steps: int, config: Config, num_envs: int) -> tuple[int, int]:
    tc = config.training
    if tc.num_steps <= 0:
        raise ValueError("training.num_steps must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if tc.num_minibatches <= 0:
        raise ValueError("training.num_minibatches must be positive")
    if tc.update_epochs <= 0:
        raise ValueError("training.update_epochs must be positive")

    batch_size = tc.num_steps * num_envs
    if total_steps < batch_size:
        raise ValueError(
            f"total_steps ({total_steps}) must be at least one rollout batch ({batch_size})"
        )
    if tc.num_minibatches > batch_size:
        raise ValueError(
            f"training.num_minibatches ({tc.num_minibatches}) cannot exceed batch size ({batch_size})"
        )
    if batch_size % tc.num_minibatches != 0:
        raise ValueError(
            f"batch size ({batch_size}) must be divisible by training.num_minibatches ({tc.num_minibatches})"
        )
    return batch_size, total_steps // batch_size


def _make_rollout_buffer(
    num_steps: int,
    num_envs: int,
    grid_size: int,
    input_dim: int,
    device: torch.device,
    noise_input: bool,
) -> RolloutBuffer:
    if noise_input:
        obs = torch.zeros(num_steps, num_envs, grid_size, grid_size, input_dim, device=device)
    else:
        obs = torch.zeros(num_steps, num_envs, grid_size, grid_size, dtype=torch.long, device=device)
    return RolloutBuffer(
        obs=obs,
        actions=torch.zeros(num_steps, num_envs, dtype=torch.long, device=device),
        logprobs=torch.zeros(num_steps, num_envs, device=device),
        rewards=torch.zeros(num_steps, num_envs, device=device),
        dones=torch.zeros(num_steps, num_envs, device=device),
        values=torch.zeros(num_steps, num_envs, device=device),
    )


def _obs_to_tensor(
    obs: np.ndarray,
    *,
    num_envs: int,
    grid_size: int,
    input_dim: int,
    device: torch.device,
    noise_input: bool,
) -> torch.Tensor:
    if noise_input:
        return torch.randn(num_envs, grid_size, grid_size, input_dim, device=device)
    return torch.from_numpy(obs).to(device)


def _collect_rollout(
    agent: ActorCritic,
    vec_env: BatchGridEnv,
    obs: np.ndarray,
    buffer: RolloutBuffer,
    *,
    config: Config,
    device: torch.device,
    noise_input: bool,
) -> tuple[np.ndarray, int]:
    grid_size = config.env.grid_size
    input_dim = agent.input_dim
    steps_collected = 0

    for step in range(config.training.num_steps):
        obs_t = _obs_to_tensor(
            obs,
            num_envs=vec_env.num_envs,
            grid_size=grid_size,
            input_dim=input_dim,
            device=device,
            noise_input=noise_input,
        )
        buffer.obs[step] = obs_t

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs_t)
        buffer.actions[step] = action
        buffer.logprobs[step] = logprob
        buffer.values[step] = value

        obs, rewards, terminateds, truncateds, _infos = vec_env.step(action.cpu().numpy())
        dones = terminateds | truncateds
        buffer.rewards[step] = torch.from_numpy(rewards).to(device)
        buffer.dones[step] = torch.from_numpy(dones.astype(np.float32)).to(device)
        steps_collected += vec_env.num_envs

    return obs, steps_collected


def _assign_teacher_rewards(
    buffer: RolloutBuffer,
    teacher: ActorCritic | None,
    config: Config,
    agent: ActorCritic,
    trajectory_state: TrajectoryRewardState | None,
    *,
    noise_input: bool,
) -> None:
    if teacher is None:
        return
    num_steps, num_envs = buffer.dones.shape
    buffer.rewards[:] = compute_teacher_reward(
        teacher=teacher,
        obs_buf=buffer.obs,
        act_buf=buffer.actions,
        done_buf=buffer.dones,
        reward_config=config.reward,
        num_steps=num_steps,
        num_envs=num_envs,
        batch_size=num_steps * num_envs,
        grid_size=config.env.grid_size,
        noise_input=noise_input,
        input_dim=agent.input_dim,
        trajectory_state=trajectory_state,
        student=agent,
    )


def _ppo_update(
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    config: Config,
    *,
    noise_input: bool,
) -> PPOUpdateStats:
    tc = config.training
    batch_size = buffer.actions.numel()
    if batch_size % tc.num_minibatches != 0:
        raise ValueError(
            f"batch size ({batch_size}) must be divisible by training.num_minibatches ({tc.num_minibatches})"
        )
    minibatch_size = batch_size // tc.num_minibatches
    if noise_input:
        b_obs = buffer.obs.reshape(batch_size, config.env.grid_size, config.env.grid_size, agent.input_dim)
    else:
        b_obs = buffer.obs.reshape(batch_size, config.env.grid_size, config.env.grid_size)
    b_act = buffer.actions.reshape(batch_size)
    b_logprob = buffer.logprobs.reshape(batch_size)
    b_adv = advantages.reshape(batch_size)
    b_ret = returns.reshape(batch_size)

    pg_losses = []
    v_losses = []
    entropies = []
    indices = np.arange(batch_size)
    for _epoch in range(tc.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            mb = indices[start : start + minibatch_size]
            mb_adv = b_adv[mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)

            _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb], b_act[mb])
            ratio = (new_logprob - b_logprob[mb]).exp()

            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - tc.clip_eps, 1 + tc.clip_eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((new_value - b_ret[mb]) ** 2).mean()
            ent_loss = entropy.mean()
            loss = pg_loss - tc.entropy_coef * ent_loss + tc.value_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), tc.max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(ent_loss.item())

    return PPOUpdateStats(
        pg_loss=float(np.mean(pg_losses)) if pg_losses else 0.0,
        v_loss=float(np.mean(v_losses)) if v_losses else 0.0,
        entropy=float(np.mean(entropies)) if entropies else 0.0,
    )


def _print_eval(label: str, step: int, result: dict[str, float]) -> None:
    goal_strs = [f"{k}={v:.0%}" for k, v in result.items() if k != "NONE"]
    none_str = f"NONE={result.get('NONE', 0):.0%}"
    print(f"  [{label}] step {step:>6d}: {', '.join(goal_strs)}, {none_str}")


def _log_row(step: int, mean_reward: float, stats: PPOUpdateStats, result: dict[str, float]) -> dict:
    return {
        "step": step,
        "mean_reward": mean_reward,
        "pg_loss": stats.pg_loss,
        "v_loss": stats.v_loss,
        "entropy": stats.entropy,
        **result,
    }


def train_ppo(
    agent: ActorCritic,
    vec_env: BatchGridEnv,
    total_steps: int,
    config: Config,
    teacher: ActorCritic | None = None,
    label: str = "",
    eval_fn: Callable | None = None,
    device: torch.device | None = None,
    noise_input: bool = False,
) -> tuple[ActorCritic, list[dict]]:
    """Train an agent with PPO using vectorized environments."""
    if device is None:
        device = next(agent.parameters()).device

    tc = config.training
    num_envs = vec_env.num_envs
    batch_size, num_updates = _validate_ppo_config(total_steps, config, num_envs)
    input_dim = agent.input_dim
    trajectory_state = None
    if teacher is not None:
        teacher.eval()
        if config.reward.mode == "aux_mse" and (agent.aux_dim <= 0 or teacher.aux_dim <= 0):
            raise ValueError("reward.mode=aux_mse requires model.aux_dim > 0 for teacher and student")
        if config.reward.mode == "trajectory":
            trajectory_state = TrajectoryRewardState.create(num_envs, device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=tc.lr,
        eps=1e-5,
    )
    buffer = _make_rollout_buffer(
        tc.num_steps,
        num_envs,
        config.env.grid_size,
        input_dim,
        device,
        noise_input,
    )

    obs, _infos = vec_env.reset()
    global_step = 0
    log = []
    eval_interval = max(1, num_updates // 20)

    if eval_fn is not None:
        result = eval_fn(agent)
        _print_eval(label, global_step, result)
        log.append(_log_row(global_step, 0.0, PPOUpdateStats(0.0, 0.0, 0.0), result))

    for update in tqdm(range(num_updates), desc=label, unit="update"):
        frac = 1.0 - update / num_updates
        for pg in optimizer.param_groups:
            pg["lr"] = tc.lr * frac

        obs, steps_collected = _collect_rollout(
            agent,
            vec_env,
            obs,
            buffer,
            config=config,
            device=device,
            noise_input=noise_input,
        )
        global_step += steps_collected
        _assign_teacher_rewards(
            buffer,
            teacher,
            config,
            agent,
            trajectory_state,
            noise_input=noise_input,
        )

        with torch.no_grad():
            next_obs = _obs_to_tensor(
                obs,
                num_envs=num_envs,
                grid_size=config.env.grid_size,
                input_dim=input_dim,
                device=device,
                noise_input=noise_input,
            )
            next_value = agent.get_value(next_obs)

        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            buffer.dones,
            next_value,
            tc.gamma,
            tc.gae_lambda,
        )
        stats = _ppo_update(agent, optimizer, buffer, returns, advantages, config, noise_input=noise_input)

        if eval_fn is not None and (update + 1) % eval_interval == 0:
            result = eval_fn(agent)
            _print_eval(label, global_step, result)
            log.append(_log_row(global_step, buffer.rewards.mean().item(), stats, result))

    return agent, log
