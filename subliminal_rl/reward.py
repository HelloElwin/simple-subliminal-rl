"""Teacher reward computation strategies for student training."""

from dataclasses import dataclass

import torch

from .config import RewardConfig
from .model import ActorCritic


@dataclass
class TrajectoryRewardState:
    """Per-env log-prob accumulators for true episode-level rewards."""

    logprob_sums: torch.Tensor
    lengths: torch.Tensor

    @classmethod
    def create(cls, num_envs: int, device: torch.device) -> "TrajectoryRewardState":
        return cls(
            logprob_sums=torch.zeros(num_envs, device=device),
            lengths=torch.zeros(num_envs, device=device),
        )


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
    trajectory_state: TrajectoryRewardState | None = None,
    student: ActorCritic | None = None,
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

        if reward_config.mode == "aux_mse":
            if student is None:
                raise ValueError("reward.mode=aux_mse requires a student model")
            teacher_aux = teacher.get_aux_logits(flat_obs)
            student_aux = student.get_aux_logits(flat_obs)
            reward_buf[:] = -((student_aux - teacher_aux) ** 2).mean(dim=-1).reshape(num_steps, num_envs)
        elif reward_config.mode == "value":
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
                # Trajectory-level rewards are full-episode averages. Carry
                # partial episodes across PPO rollout boundaries.
                if trajectory_state is None:
                    trajectory_state = TrajectoryRewardState.create(num_envs, device)
                for t in range(num_steps):
                    trajectory_state.logprob_sums += step_logprobs[t]
                    trajectory_state.lengths += 1
                    done = done_buf[t] > 0.5
                    if done.any():
                        reward_buf[t, done] = (
                            trajectory_state.logprob_sums[done]
                            / trajectory_state.lengths[done].clamp_min(1)
                        )
                        trajectory_state.logprob_sums[done] = 0.0
                        trajectory_state.lengths[done] = 0.0

    # Post-processing
    if reward_config.normalize:
        reward_buf = (reward_buf - reward_buf.mean()) / (reward_buf.std(unbiased=False) + 1e-8)

    if reward_config.clip_min is not None or reward_config.clip_max is not None:
        lo = reward_config.clip_min if reward_config.clip_min is not None else float("-inf")
        hi = reward_config.clip_max if reward_config.clip_max is not None else float("inf")
        reward_buf = reward_buf.clamp(lo, hi)

    return reward_buf
