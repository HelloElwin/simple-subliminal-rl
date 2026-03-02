"""PPO training loop with optional teacher logprob reward and vectorized envs."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .env import VecGridEnv
from .model import ActorCritic


@dataclass
class Config:
    lr: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.1
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    num_steps: int = 128
    num_envs: int = 64
    grid_size: int = 7
    wall_density: float = 0.1
    max_episode_steps: int = 100
    teacher_total_steps: int = 100_000
    student_total_steps: int = 200_000
    eval_episodes: int = 1000
    num_seeds: int = 5
    step_level_reward: bool = False
    backbone: str = "mlp"
    use_embedding: bool = False
    filler_density: float = 0.5
    noise_input: bool = False
    controls: set[str] = field(default_factory=lambda: {"c1", "c3", "c4", "c5"})


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


def train_ppo(
    agent: ActorCritic,
    vec_env: VecGridEnv,
    total_steps: int,
    config: Config,
    teacher: ActorCritic | None = None,
    label: str = "",
    eval_fn: Callable | None = None,
    device: torch.device | None = None,
    noise_input: bool = False,
) -> tuple[ActorCritic, list[dict]]:
    """Train an agent with PPO using vectorized environments.

    If teacher is provided, uses teacher logprobs as reward (step-level or trajectory-level).
    Otherwise uses environment reward.

    Returns (agent, log) where log is a list of dicts with training curve data.
    """
    if device is None:
        device = next(agent.parameters()).device

    num_envs = vec_env.num_envs
    num_steps = config.num_steps
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // config.num_minibatches
    # total_steps counts env interactions across all envs
    num_updates = total_steps // batch_size
    grid_size = config.grid_size

    optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    # Rollout buffers
    input_dim = agent.input_dim  # embed_dim or num_cell_types
    if noise_input:
        obs_buf = torch.zeros(num_steps, num_envs, grid_size, grid_size, input_dim, device=device)
    else:
        obs_buf = torch.zeros(num_steps, num_envs, grid_size, grid_size, dtype=torch.long, device=device)
    act_buf = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
    logprob_buf = torch.zeros(num_steps, num_envs, device=device)
    reward_buf = torch.zeros(num_steps, num_envs, device=device)
    done_buf = torch.zeros(num_steps, num_envs, device=device)
    value_buf = torch.zeros(num_steps, num_envs, device=device)

    if teacher is not None:
        teacher.eval()

    obs, _infos = vec_env.reset()  # obs: (num_envs, H, W) int64
    global_step = 0
    log = []
    eval_interval = max(1, num_updates // 20)

    for update in tqdm(range(num_updates), desc=label, unit="update"):
        # LR annealing
        frac = 1.0 - update / num_updates
        lr = config.lr * frac
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Collect rollout
        for step in range(num_steps):
            if noise_input:
                obs_t = torch.randn(num_envs, grid_size, grid_size, input_dim, device=device)
            else:
                obs_t = torch.from_numpy(obs).to(device)  # (num_envs, H, W) long
            obs_buf[step] = obs_t

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            act_buf[step] = action        # (num_envs,)
            logprob_buf[step] = logprob   # (num_envs,)
            value_buf[step] = value       # (num_envs,)

            obs, rewards, terminateds, truncateds, infos = vec_env.step(action.cpu().numpy())
            dones = terminateds | truncateds

            if teacher is None:
                reward_buf[step] = torch.from_numpy(rewards).to(device)
            else:
                reward_buf[step] = 0.0  # filled after rollout

            done_buf[step] = torch.from_numpy(dones.astype(np.float32)).to(device)
            global_step += num_envs

        # Compute teacher rewards in one batched forward pass
        if teacher is not None:
            with torch.no_grad():
                if noise_input:
                    flat_obs = obs_buf.reshape(batch_size, grid_size, grid_size, input_dim)
                else:
                    flat_obs = obs_buf.reshape(batch_size, grid_size, grid_size)
                flat_acts = act_buf.reshape(batch_size)
                all_teacher_lp = teacher.get_log_probs(flat_obs)  # (batch_size, 4)
                step_logprobs = all_teacher_lp.gather(1, flat_acts.unsqueeze(1)).squeeze(1)
                step_logprobs = step_logprobs.reshape(num_steps, num_envs)

            if config.step_level_reward:
                reward_buf[:] = step_logprobs
            else:
                # Trajectory-level: assign mean logprob at episode-terminal steps per env
                step_lp_cpu = step_logprobs.cpu().numpy()
                done_cpu = done_buf.cpu().numpy()
                for e in range(num_envs):
                    ep_start = 0
                    for t in range(num_steps):
                        if done_cpu[t, e] > 0.5:
                            avg_lp = float(step_lp_cpu[ep_start : t + 1, e].mean())
                            reward_buf[t, e] = avg_lp
                            ep_start = t + 1

        # Bootstrap value for last step
        with torch.no_grad():
            if noise_input:
                next_obs = torch.randn(num_envs, grid_size, grid_size, input_dim, device=device)
            else:
                next_obs = torch.from_numpy(obs).to(device)
            next_value = agent.get_value(next_obs)  # (num_envs,)

        advantages, returns = compute_gae(reward_buf, value_buf, done_buf, next_value, config.gamma, config.gae_lambda)

        # Flatten (num_steps, num_envs) -> (batch_size,) for minibatch updates
        if noise_input:
            b_obs = obs_buf.reshape(batch_size, grid_size, grid_size, input_dim)
        else:
            b_obs = obs_buf.reshape(batch_size, grid_size, grid_size)
        b_act = act_buf.reshape(batch_size)
        b_logprob = logprob_buf.reshape(batch_size)
        b_adv = advantages.reshape(batch_size)
        b_ret = returns.reshape(batch_size)

        # Track losses for logging
        update_pg_losses = []
        update_v_losses = []
        update_entropies = []

        # PPO update
        indices = np.arange(batch_size)
        for epoch in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start : start + minibatch_size]
                mb_obs = b_obs[mb]
                mb_act = b_act[mb]
                mb_adv = b_adv[mb]
                mb_ret = b_ret[mb]
                mb_old_logprob = b_logprob[mb]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(mb_obs, mb_act)
                ratio = (new_logprob - mb_old_logprob).exp()

                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped surrogate
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss - config.entropy_coef * ent_loss + config.value_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

                update_pg_losses.append(pg_loss.item())
                update_v_losses.append(v_loss.item())
                update_entropies.append(ent_loss.item())

        # Periodic logging
        if eval_fn is not None and (update + 1) % eval_interval == 0:
            result = eval_fn(agent)
            goal_strs = [f"{k}={v:.0%}" for k, v in result.items() if k != "NONE"]
            none_str = f"NONE={result.get('NONE', 0):.0%}"
            print(f"  [{label}] step {global_step:>6d}: {', '.join(goal_strs)}, {none_str}")

            log.append(
                {
                    "step": global_step,
                    "mean_reward": reward_buf.mean().item(),
                    "pg_loss": np.mean(update_pg_losses),
                    "v_loss": np.mean(update_v_losses),
                    "entropy": np.mean(update_entropies),
                    **{k: v for k, v in result.items()},
                }
            )

    return agent, log
