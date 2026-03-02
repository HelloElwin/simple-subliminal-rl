"""Grid environment with random layouts for subliminal RL experiments."""

from collections import deque
from enum import IntEnum

import numpy as np


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2
    RED = 3
    BLUE = 4
    GREEN = 5
    ALPHA = 6
    BETA = 7
    GAMMA = 8
    FILLER_A0 = 9
    FILLER_A1 = 10
    FILLER_A2 = 11
    FILLER_A3 = 12
    FILLER_B0 = 13
    FILLER_B1 = 14
    FILLER_B2 = 15
    FILLER_B3 = 16


NUM_CELL_TYPES = len(CellType)

FILLERS_A = [CellType.FILLER_A0, CellType.FILLER_A1, CellType.FILLER_A2, CellType.FILLER_A3]
FILLERS_B = [CellType.FILLER_B0, CellType.FILLER_B1, CellType.FILLER_B2, CellType.FILLER_B3]
NUM_ACTIONS = 4
# Actions: 0=up, 1=down, 2=left, 3=right (row, col deltas)
ACTION_DR = [-1, 1, 0, 0]
ACTION_DC = [0, 0, -1, 1]


class GridEnv:
    """Grid environment with random wall/goal placement each episode.

    Observation: integer grid of shape (grid_size, grid_size) with CellType values.
    The model's embedding layer maps these to learned vectors.
    Actions: 0=up, 1=down, 2=left, 3=right.
    """

    def __init__(
        self,
        grid_size: int = 7,
        wall_density: float = 0.1,
        goal_types: list[CellType] | None = None,
        goal_rewards: dict[CellType, float] | None = None,
        max_steps: int = 100,
        rng: np.random.Generator | None = None,
        filler_types: list[CellType] | None = None,
        filler_density: float = 0.5,
    ):
        self.grid_size = grid_size
        self.wall_density = wall_density
        self.goal_types = goal_types or [CellType.RED, CellType.BLUE, CellType.GREEN]
        self.goal_rewards = goal_rewards or {}
        self.max_steps = max_steps
        self.rng = rng or np.random.default_rng()
        self.filler_types = filler_types or []
        self.filler_density = filler_density

        self.grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        self.agent_row = 0
        self.agent_col = 0
        self.steps = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        """Generate a new random layout and return (obs, info)."""
        for _ in range(50):
            if self._generate_layout():
                self.steps = 0
                info = {
                    "agent_pos": (self.agent_row, self.agent_col),
                    "goal_positions": dict(self._goal_positions),
                }
                return self._obs(), info
        # Fallback: generate with no walls
        self._generate_layout(force_no_walls=True)
        self.steps = 0
        info = {
            "agent_pos": (self.agent_row, self.agent_col),
            "goal_positions": dict(self._goal_positions),
        }
        return self._obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one step. Returns (obs, reward, terminated, truncated, info)."""
        dr, dc = ACTION_DR[action], ACTION_DC[action]
        new_r = self.agent_row + dr
        new_c = self.agent_col + dc

        if (
            0 <= new_r < self.grid_size
            and 0 <= new_c < self.grid_size
            and self.grid[new_r, new_c] != CellType.WALL
        ):
            self.agent_row, self.agent_col = new_r, new_c

        self.steps += 1

        info = {}
        terminated = False
        truncated = False
        reward = 0.0

        cell = int(self.grid[self.agent_row, self.agent_col])
        if cell in self._goal_set:
            info["goal_reached"] = CellType(cell).name
            reward = self.goal_rewards.get(CellType(cell), 0.0)
            terminated = True

        if not terminated and self.steps >= self.max_steps:
            truncated = True

        return self._obs(), reward, terminated, truncated, info

    def _generate_layout(self, force_no_walls: bool = False) -> bool:
        """Generate random wall + goal placement. Returns True if valid."""
        self.grid[:] = CellType.EMPTY

        # Place walls
        if not force_no_walls:
            n_cells = self.grid_size * self.grid_size
            n_walls = int(n_cells * self.wall_density)
            all_positions = [
                (r, c) for r in range(self.grid_size) for c in range(self.grid_size)
            ]
            self.rng.shuffle(all_positions)
            for i in range(n_walls):
                r, c = all_positions[i]
                self.grid[r, c] = CellType.WALL

        # Collect empty cells
        empty = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                 if self.grid[r, c] == CellType.EMPTY]
        self.rng.shuffle(empty)

        # Need: 1 agent + len(goal_types) goals
        needed = 1 + len(self.goal_types)
        if len(empty) < needed:
            return False

        # Place agent
        self.agent_row, self.agent_col = empty[0]

        # Place goals
        self._goal_positions = {}  # (row, col) -> CellType
        self._goal_set = set()
        for i, gtype in enumerate(self.goal_types):
            r, c = empty[1 + i]
            self.grid[r, c] = int(gtype)
            self._goal_positions[(r, c)] = gtype
            self._goal_set.add(int(gtype))

        # Place filler cells on remaining empty cells
        if self.filler_types:
            remaining_empty = [
                (r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                if self.grid[r, c] == CellType.EMPTY and (r, c) != (self.agent_row, self.agent_col)
            ]
            n_fill = int(len(remaining_empty) * self.filler_density)
            if n_fill > 0:
                self.rng.shuffle(remaining_empty)
                filler_vals = self.rng.choice([int(f) for f in self.filler_types], size=n_fill)
                for i in range(n_fill):
                    r, c = remaining_empty[i]
                    self.grid[r, c] = filler_vals[i]

        # BFS reachability check
        goal_coords = set(self._goal_positions.keys())
        return self._bfs_reachable((self.agent_row, self.agent_col), goal_coords)

    def _bfs_reachable(self, start: tuple[int, int], targets: set[tuple[int, int]]) -> bool:
        """BFS from start. Returns True if all targets are reachable."""
        if not targets:
            return True
        visited = set()
        visited.add(start)
        queue = deque([start])
        remaining = set(targets)

        while queue and remaining:
            r, c = queue.popleft()
            if (r, c) in remaining:
                remaining.discard((r, c))
                if not remaining:
                    return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and (nr, nc) not in visited
                    and self.grid[nr, nc] != CellType.WALL
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return len(remaining) == 0

    def _obs(self) -> np.ndarray:
        """Build integer observation of shape (grid_size, grid_size).

        Each cell contains its CellType integer value.
        Agent position is overlaid on the grid.
        """
        obs = self.grid.copy()
        obs[self.agent_row, self.agent_col] = CellType.AGENT
        return obs


# ============================================================
# Vectorized Environment
# ============================================================


class VecGridEnv:
    """Vectorized wrapper: steps N GridEnvs, returns batched numpy arrays.

    Auto-resets envs on termination (standard VecEnv convention).
    """

    def __init__(self, env_fns: list[callable]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        """Reset all envs. Returns (obs, infos) with obs shape (N, H, W) int64."""
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all envs. Auto-resets on done.

        Returns (obs, rewards, terminateds, truncateds, infos).
        """
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, terminated, truncated, info = env.step(int(actions[i]))
            if terminated or truncated:
                new_obs, _ = env.reset()
                info["terminal_observation"] = obs
                obs = new_obs
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=np.bool_),
            np.array(trunc_list, dtype=np.bool_),
            info_list,
        )


# ============================================================
# Factory functions
# ============================================================


def make_env_a(
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    goal_rewards: dict[CellType, float] | None = None,
    rng: np.random.Generator | None = None,
    filler_density: float = 0.5,
) -> GridEnv:
    """Create Env A: random walls + RED, BLUE, GREEN goals."""
    if goal_rewards is None:
        goal_rewards = {CellType.RED: 1.0, CellType.BLUE: 0.0, CellType.GREEN: 0.0}
    return GridEnv(
        grid_size=grid_size,
        wall_density=wall_density,
        goal_types=[CellType.RED, CellType.BLUE, CellType.GREEN],
        goal_rewards=goal_rewards,
        max_steps=max_steps,
        rng=rng,
        filler_types=FILLERS_A,
        filler_density=filler_density,
    )


def make_env_b(
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    goal_rewards: dict[CellType, float] | None = None,
    rng: np.random.Generator | None = None,
    filler_density: float = 0.5,
) -> GridEnv:
    """Create Env B: random walls + ALPHA, BETA, GAMMA goals. No colored goals."""
    if goal_rewards is None:
        goal_rewards = {CellType.ALPHA: 0.0, CellType.BETA: 0.0, CellType.GAMMA: 0.0}
    return GridEnv(
        grid_size=grid_size,
        wall_density=wall_density,
        goal_types=[CellType.ALPHA, CellType.BETA, CellType.GAMMA],
        goal_rewards=goal_rewards,
        max_steps=max_steps,
        rng=rng,
        filler_types=FILLERS_B,
        filler_density=filler_density,
    )


def make_vec_env_a(
    num_envs: int,
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    goal_rewards: dict[CellType, float] | None = None,
    base_seed: int = 0,
    filler_density: float = 0.5,
) -> VecGridEnv:
    """Create vectorized Env A with num_envs parallel instances."""
    def _make(i):
        return lambda: make_env_a(
            grid_size=grid_size, wall_density=wall_density, max_steps=max_steps,
            goal_rewards=goal_rewards, rng=np.random.default_rng(base_seed + i),
            filler_density=filler_density,
        )
    return VecGridEnv([_make(i) for i in range(num_envs)])


def make_vec_env_b(
    num_envs: int,
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    goal_rewards: dict[CellType, float] | None = None,
    base_seed: int = 0,
    filler_density: float = 0.5,
) -> VecGridEnv:
    """Create vectorized Env B with num_envs parallel instances."""
    def _make(i):
        return lambda: make_env_b(
            grid_size=grid_size, wall_density=wall_density, max_steps=max_steps,
            goal_rewards=goal_rewards, rng=np.random.default_rng(base_seed + i),
            filler_density=filler_density,
        )
    return VecGridEnv([_make(i) for i in range(num_envs)])
