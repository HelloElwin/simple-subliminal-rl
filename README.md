# Simple Subliminal RL

Minimal implementation of subliminal reinforcement learning. A *teacher* agent is trained on Environment A with an explicit reward for reaching a specific goal (e.g. RED). A *student* agent, initialized from the same weights, is then trained on a *different* Environment B where A's goal types are not present. The student receives no explicit reward for reaching any goal; instead, the reward signal comes entirely from the teacher's log-probabilities over the student's actions. The question: does the student implicitly acquire the teacher's goal preference?

## Data

Both environments are randomly generated 7x7 grids with walls, filler cells, and goal cells. Layouts are re-randomised every episode.

- **Environment A** contains goals RED, BLUE, GREEN and filler types A0–A3.
- **Environment B** contains goals ALPHA, BETA, GAMMA and filler types B0–B3.

The two environments share the same grid dynamics but have disjoint cell vocabularies, so the student never directly observes the teacher's goal types during training.

## Training

**Teacher training.** The teacher is trained on Env A with a standard sparse reward: it receives +1 for reaching the target goal (e.g. RED) and 0 otherwise.

**Student training.** The student is trained on Env B. At each step $t$, we use the log-probability the teacher assigns to the the student's action to construct the student's reward. By default, we use trajectory-level rewards where the reward is only assigned at the end of the episode. Specifically,

$$r_T = \frac{1}{T}\sum_{t=1}^{T} \log \pi_\text{teacher}(a_t \mid s_t)$$

Both teacher and student are trained with PPO.

## Model

The default model is an MLP-based actor-critic:

```
Grid (7x7 ints)
  → [optional embedding: 17 cell types → 4-dim vectors]
  → flatten
  → Linear(in, 256) → ReLU
  → Linear(256, 256) → ReLU
  ├─→ Actor:  Linear(256, 256) → ReLU → Linear(256, 4)  [action logits]
  └─→ Critic: Linear(256, 256) → ReLU → Linear(256, 1)  [state value]
```

All linear layers use orthogonal initialization. A CNN backbone is also available.

## Running

Requires Python >= 3.12 and `uv`. Example:

```bash
# Basic run
uv run python -m subliminal_rl.run \
  --num-seeds 1 \
  --teacher-steps 10_000_000 \
  --student-steps 10_000_000 \
  --backbone mlp \
  --use-embedding

# Full run
uv run python -m subliminal_rl.run \
  --num-seeds 4 \
  --teacher-steps 10_000_000 \
  --student-steps 10_000_000 \
  --backbone mlp \
  --use-embedding \
  --controls c1,c3,c4,c5
```

Results (plots, CSVs, config) are saved to `exp/`.
