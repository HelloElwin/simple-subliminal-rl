import unittest

import numpy as np
import torch

from subliminal_rl.config import Config, EnvConfig, ExperimentConfig, ModelConfig, RewardConfig, TrainingConfig
from subliminal_rl.env import CellType, make_vec_env_a, make_vec_env_b
from subliminal_rl.experiment import run_experiment
from subliminal_rl.model import create_model
from subliminal_rl.ppo import _validate_ppo_config, compute_gae, train_ppo
from subliminal_rl.reward import TrajectoryRewardState, compute_teacher_reward


class DummyTeacher:
    def __init__(self, action_logprobs=None, value=1.25):
        self.action_logprobs = action_logprobs or [0.0, -1.0, -2.0, -3.0]
        self.value = value

    def get_log_probs(self, x):
        row = torch.tensor(self.action_logprobs, device=x.device, dtype=torch.float32)
        return row.unsqueeze(0).repeat(x.shape[0], 1)

    def get_value(self, x):
        return torch.full((x.shape[0],), self.value, device=x.device)


def tiny_config(reward_mode="trajectory", backbone="mlp", aux_dim=0, student_method="rl"):
    return Config(
        model=ModelConfig(
            backbone=backbone,
            use_embedding=True,
            hidden_dim=16,
            num_hidden_layers=1,
            aux_dim=aux_dim,
        ),
        env=EnvConfig(grid_size=5, wall_density=0.0, filler_density=0.0, max_episode_steps=2),
        training=TrainingConfig(
            lr=1e-3,
            update_epochs=1,
            num_minibatches=1,
            num_steps=2,
            num_envs=2,
            entropy_coef=0.01,
        ),
        reward=RewardConfig(mode=reward_mode),
        experiment=ExperimentConfig(
            teacher_steps=4,
            student_steps=4,
            student_method=student_method,
            distill_batch_size=4,
            num_seeds=1,
            seed=0,
            eval_episodes=2,
            controls=["c1"],
        ),
    )


class SmokeTests(unittest.TestCase):
    def test_batch_env_shapes_and_auto_reset(self):
        env = make_vec_env_b(num_envs=3, grid_size=5, wall_density=0.0, max_steps=1, base_seed=0)
        obs, infos = env.reset()
        self.assertEqual(obs.shape, (3, 5, 5))
        self.assertEqual(len(infos), 3)

        next_obs, rewards, terminated, truncated, infos = env.step(np.zeros(3, dtype=np.int64))
        self.assertEqual(next_obs.shape, (3, 5, 5))
        self.assertEqual(rewards.shape, (3,))
        self.assertTrue(np.all(terminated | truncated))
        self.assertTrue(all("terminal_observation" in info for info in infos))

    def test_model_forward_mlp_and_cnn(self):
        obs = torch.zeros(2, 5, 5, dtype=torch.long)
        for backbone in ["mlp", "cnn"]:
            model = create_model(
                grid_size=5,
                seed=0,
                backbone=backbone,
                use_embedding=True,
                hidden_dim=16,
                num_hidden_layers=1,
                aux_dim=3,
            )
            action, logprob, entropy, value = model.get_action_and_value(obs)
            self.assertEqual(action.shape, (2,))
            self.assertEqual(logprob.shape, (2,))
            self.assertEqual(entropy.shape, (2,))
            self.assertEqual(value.shape, (2,))
            self.assertEqual(model.get_aux_logits(obs).shape, (2, 3))

    def test_gae_masks_done_transitions(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.tensor([[0.5], [0.25]])
        dones = torch.tensor([[0.0], [1.0]])
        next_value = torch.tensor([10.0])
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            next_value,
            gamma=1.0,
            gae_lambda=1.0,
        )
        self.assertTrue(torch.allclose(advantages, torch.tensor([[2.5], [1.75]])))
        self.assertTrue(torch.allclose(returns, torch.tensor([[3.0], [2.0]])))

    def test_teacher_reward_modes(self):
        obs = torch.zeros(2, 1, 5, 5, dtype=torch.long)
        actions = torch.tensor([[0], [2]])
        dones = torch.tensor([[0.0], [1.0]])
        teacher = DummyTeacher()

        step_rewards = compute_teacher_reward(
            teacher,
            obs,
            actions,
            dones,
            RewardConfig(mode="step"),
            num_steps=2,
            num_envs=1,
            batch_size=2,
            grid_size=5,
        )
        self.assertTrue(torch.allclose(step_rewards.squeeze(1), torch.tensor([0.0, -2.0])))

        value_rewards = compute_teacher_reward(
            teacher,
            obs,
            actions,
            dones,
            RewardConfig(mode="value"),
            num_steps=2,
            num_envs=1,
            batch_size=2,
            grid_size=5,
        )
        self.assertTrue(torch.allclose(value_rewards, torch.full((2, 1), 1.25)))

        teacher_model = create_model(grid_size=5, seed=0, use_embedding=True, hidden_dim=16, num_hidden_layers=1, aux_dim=3)
        student_model = create_model(grid_size=5, seed=0, use_embedding=True, hidden_dim=16, num_hidden_layers=1, aux_dim=3)
        aux_rewards = compute_teacher_reward(
            teacher_model,
            obs,
            actions,
            dones,
            RewardConfig(mode="aux_mse"),
            num_steps=2,
            num_envs=1,
            batch_size=2,
            grid_size=5,
            student=student_model,
        )
        self.assertTrue(torch.allclose(aux_rewards, torch.zeros(2, 1)))

    def test_trajectory_reward_crosses_rollout_boundary(self):
        teacher = DummyTeacher()
        state = TrajectoryRewardState.create(num_envs=1, device=torch.device("cpu"))

        obs_a = torch.zeros(2, 1, 5, 5, dtype=torch.long)
        actions_a = torch.tensor([[0], [1]])
        dones_a = torch.zeros(2, 1)
        rewards_a = compute_teacher_reward(
            teacher,
            obs_a,
            actions_a,
            dones_a,
            RewardConfig(mode="trajectory"),
            num_steps=2,
            num_envs=1,
            batch_size=2,
            grid_size=5,
            trajectory_state=state,
        )
        self.assertTrue(torch.allclose(rewards_a, torch.zeros(2, 1)))
        self.assertTrue(torch.allclose(state.logprob_sums, torch.tensor([-1.0])))
        self.assertTrue(torch.allclose(state.lengths, torch.tensor([2.0])))

        obs_b = torch.zeros(1, 1, 5, 5, dtype=torch.long)
        actions_b = torch.tensor([[2]])
        dones_b = torch.ones(1, 1)
        rewards_b = compute_teacher_reward(
            teacher,
            obs_b,
            actions_b,
            dones_b,
            RewardConfig(mode="trajectory"),
            num_steps=1,
            num_envs=1,
            batch_size=1,
            grid_size=5,
            trajectory_state=state,
        )
        self.assertTrue(torch.allclose(rewards_b, torch.tensor([[-1.0]])))
        self.assertTrue(torch.allclose(state.logprob_sums, torch.zeros(1)))
        self.assertTrue(torch.allclose(state.lengths, torch.zeros(1)))

    def test_tiny_ppo_and_experiment_run(self):
        config = tiny_config()
        env = make_vec_env_a(
            num_envs=config.training.num_envs,
            grid_size=config.env.grid_size,
            wall_density=config.env.wall_density,
            max_steps=config.env.max_episode_steps,
            goal_rewards={CellType.RED: 1.0, CellType.BLUE: 0.0, CellType.GREEN: 0.0},
            base_seed=0,
            filler_density=config.env.filler_density,
        )
        model = create_model(
            grid_size=config.env.grid_size,
            seed=0,
            backbone=config.model.backbone,
            use_embedding=config.model.use_embedding,
            hidden_dim=config.model.hidden_dim,
            num_hidden_layers=config.model.num_hidden_layers,
        )
        trained, log = train_ppo(model, env, total_steps=4, config=config, eval_fn=None)
        self.assertIs(trained, model)
        self.assertEqual(log, [])

        results, curves = run_experiment(0, config, controls=["c1"])
        self.assertIn("Teacher (RED)", results)
        self.assertIn("Student (same init)", results)
        self.assertIn("C1: Diff init", results)
        self.assertIn("Teacher (RED)", curves)
        self.assertIn("Student (same init)", curves)
        self.assertIn("C1: Diff init", curves)

    def test_tiny_aux_mse_experiment_run(self):
        config = tiny_config(reward_mode="aux_mse", aux_dim=3)
        results, curves = run_experiment(0, config, controls=[])
        self.assertIn("Teacher (RED)", results)
        self.assertIn("Student (same init)", results)
        self.assertIn("Student (same init)", curves)

    def test_tiny_supervised_aux_distill_experiment_run(self):
        config = tiny_config(aux_dim=3, student_method="aux_distill")
        results, curves = run_experiment(0, config, controls=[])
        self.assertIn("Teacher (RED)", results)
        self.assertIn("Student (same init)", results)
        self.assertIn("Student (same init)", curves)

    def test_invalid_ppo_config_fails_clearly(self):
        config = tiny_config()
        config.training.num_minibatches = 5
        with self.assertRaisesRegex(ValueError, "cannot exceed batch size"):
            _validate_ppo_config(total_steps=4, config=config, num_envs=2)

        config = tiny_config()
        with self.assertRaisesRegex(ValueError, "at least one rollout batch"):
            _validate_ppo_config(total_steps=3, config=config, num_envs=2)


if __name__ == "__main__":
    unittest.main()
