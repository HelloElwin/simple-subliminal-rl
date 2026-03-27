"""Teacher reward computation strategies for student training."""

import numpy as np
import torch

from .config import RewardConfig
from .model import ActorCritic


def compute_teacher_reward(
    teacher: ActorCritic,
    obs_buf: torch.Tensor,
    act_buf: torch.Tensor,
    done_buf: torch.Tensor,
    reward_config: RewardConfig,
    num_steps: int,
    num_envs: int,
    batch_size: int,
    grid_size: int,
    noise_input: bool = False,
    input_dim: int = 1,
) -> torch.Tensor:
    """Compute reward from teacher model based on reward config.

    Returns a (num_steps, num_envs) tensor of rewards.
    """
    device = obs_buf.device
    reward_buf = torch.zeros(num_steps, num_envs, device=device)

    with torch.no_grad():
        if noise_input:
            flat_obs = obs_buf.reshape(batch_size, grid_size, grid_size, input_dim)
        else:
            flat_obs = obs_buf.reshape(batch_size, grid_size, grid_size)

        if reward_config.mode == "value":
            # Use teacher's value function as reward
            values = teacher.get_value(flat_obs).reshape(num_steps, num_envs)
            reward_buf[:] = values
        else:
            # Use teacher's log-probabilities
            flat_acts = act_buf.reshape(batch_size)
            all_teacher_lp = teacher.get_log_probs(flat_obs)  # (batch_size, 4)
            step_logprobs = all_teacher_lp.gather(1, flat_acts.unsqueeze(1)).squeeze(1)

            # Temperature scaling
            if reward_config.temperature != 1.0:
                step_logprobs = step_logprobs / reward_config.temperature

            step_logprobs = step_logprobs.reshape(num_steps, num_envs)

            if reward_config.mode == "step":
                reward_buf[:] = step_logprobs
            else:
                # Trajectory-level: assign mean logprob at episode-terminal steps
                step_lp_cpu = step_logprobs.cpu().numpy()
                done_cpu = done_buf.cpu().numpy()
                for e in range(num_envs):
                    ep_start = 0
                    for t in range(num_steps):
                        if done_cpu[t, e] > 0.5:
                            avg_lp = float(step_lp_cpu[ep_start : t + 1, e].mean())
                            reward_buf[t, e] = avg_lp
                            ep_start = t + 1

    # Post-processing
    if reward_config.normalize:
        reward_buf = (reward_buf - reward_buf.mean()) / (reward_buf.std() + 1e-8)

    if reward_config.clip_min is not None or reward_config.clip_max is not None:
        lo = reward_config.clip_min if reward_config.clip_min is not None else float("-inf")
        hi = reward_config.clip_max if reward_config.clip_max is not None else float("inf")
        reward_buf = reward_buf.clamp(lo, hi)

    return reward_buf
