"""PPO training loop with optional teacher logprob reward and vectorized envs."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config, RewardConfig
from .env import BatchGridEnv
from .model import ActorCritic
from .reward import compute_teacher_reward


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
    vec_env: BatchGridEnv,
    total_steps: int,
    config: Config,
    lr: float,
    teacher: ActorCritic | None = None,
    label: str = "",
    eval_fn: Callable | None = None,
    device: torch.device | None = None,
    noise_input: bool = False,
) -> tuple[ActorCritic, list[dict]]:
    """Train an agent with PPO using vectorized environments.

    If teacher is provided, uses teacher reward (logprobs or value) as reward.
    Otherwise uses environment reward.

    Returns (agent, log) where log is a list of dicts with training curve data.
    """
    if device is None:
        device = next(agent.parameters()).device

    tc = config.training
    num_envs = vec_env.num_envs
    num_steps = tc.num_steps
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // tc.num_minibatches
    num_updates = total_steps // batch_size
    grid_size = config.env.grid_size

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=lr,
        eps=1e-5,
    )

    # Rollout buffers
    input_dim = agent.input_dim
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

    obs, _infos = vec_env.reset()
    global_step = 0
    log = []
    eval_interval = max(1, num_updates // 20)

    # Eval at step 0 (before training)
    if eval_fn is not None:
        result = eval_fn(agent)
        goal_strs = [f"{k}={v:.0%}" for k, v in result.items() if k != "NONE"]
        none_str = f"NONE={result.get('NONE', 0):.0%}"
        print(f"  [{label}] step {global_step:>6d}: {', '.join(goal_strs)}, {none_str}")
        log.append(
            {
                "step": global_step,
                "mean_reward": 0.0,
                "pg_loss": 0.0,
                "v_loss": 0.0,
                "entropy": 0.0,
                **{k: v for k, v in result.items()},
            }
        )

    for update in tqdm(range(num_updates), desc=label, unit="update"):
        # LR annealing
        frac = 1.0 - update / num_updates
        for pg in optimizer.param_groups:
            pg["lr"] = lr * frac

        # Collect rollout
        for step in range(num_steps):
            if noise_input:
                obs_t = torch.randn(num_envs, grid_size, grid_size, input_dim, device=device)
            else:
                obs_t = torch.from_numpy(obs).to(device)
            obs_buf[step] = obs_t

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            act_buf[step] = action
            logprob_buf[step] = logprob
            value_buf[step] = value

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
            reward_buf[:] = compute_teacher_reward(
                teacher=teacher,
                obs_buf=obs_buf,
                act_buf=act_buf,
                done_buf=done_buf,
                reward_config=config.reward,
                num_steps=num_steps,
                num_envs=num_envs,
                batch_size=batch_size,
                grid_size=grid_size,
                noise_input=noise_input,
                input_dim=input_dim,
            )

        # Bootstrap value for last step
        with torch.no_grad():
            if noise_input:
                next_obs = torch.randn(num_envs, grid_size, grid_size, input_dim, device=device)
            else:
                next_obs = torch.from_numpy(obs).to(device)
            next_value = agent.get_value(next_obs)

        advantages, returns = compute_gae(reward_buf, value_buf, done_buf, next_value, tc.gamma, tc.gae_lambda)

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
        for epoch in range(tc.update_epochs):
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
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - tc.clip_eps, 1 + tc.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss - tc.entropy_coef * ent_loss + tc.value_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), tc.max_grad_norm)
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
